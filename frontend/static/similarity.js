/* =================================================================
   SIMILARITY.JS — Upload + similarity search + carousel display
   ================================================================= */

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const uploadZone     = $("#uploadZone");
const fileInput      = $("#fileInput");
const queryPreview   = $("#queryPreview");
const queryImage     = $("#queryImage");
const queryInfo      = $("#queryInfo");
const classResult    = $("#classificationResult");
const classLabel     = $("#classLabel");
const classConfidence = $("#classConfidence");
const classTime      = $("#classTime");
const metricInfo     = $("#metricInfo");
const loadingState   = $("#loadingState");
const emptyState     = $("#emptyState");
const carouselWrapper = $("#carouselWrapper");
const carousel       = $("#carousel");
const btnPrev        = $("#btnPrev");
const btnNext        = $("#btnNext");
const imageOverlay   = $("#imageOverlay");
const overlayImage   = $("#overlayImage");
const overlayInfo    = $("#overlayInfo");
const overlayClose   = $("#overlayClose");

// ---------------------------------------------------------------------------
// Backend health check
// ---------------------------------------------------------------------------
async function checkBackend() {
    const dot  = $(".status-dot");
    const text = $(".status-text");
    try {
        const res = await fetch("/api/health", { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            dot.classList.remove("offline");
            text.textContent = "Backend connecté";
            return true;
        }
    } catch (e) { /* offline */ }
    dot.classList.add("offline");
    text.textContent = "Backend hors ligne";
    return false;
}
checkBackend();
setInterval(checkBackend, 15000);

// ---------------------------------------------------------------------------
// Upload handling
// ---------------------------------------------------------------------------
uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// Drag & Drop
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith("image/")) handleFile(file);
    }
});

// ---------------------------------------------------------------------------
// Main flow
// ---------------------------------------------------------------------------
async function handleFile(file) {
    // Show query image
    const objectUrl = URL.createObjectURL(file);
    queryImage.src = objectUrl;
    queryInfo.textContent = file.name;
    queryPreview.style.display = "block";

    // Reset results
    classResult.style.display = "none";
    metricInfo.style.display = "none";
    carouselWrapper.style.display = "none";
    emptyState.style.display = "none";
    loadingState.style.display = "flex";

    // Animate query image in
    gsap.fromTo(queryPreview, { opacity: 0, scale: 0.9 }, { opacity: 1, scale: 1, duration: 0.4, ease: "back.out(1.4)" });

    try {
        const result = await searchSimilar(file);
        showResults(result);
    } catch (e) {
        loadingState.style.display = "none";
        emptyState.style.display = "flex";
        emptyState.querySelector("p").textContent = `Erreur: ${e.message}`;
    }
}

// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------
async function searchSimilar(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/similar", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
    }

    return await response.json();
}

// ---------------------------------------------------------------------------
// Display results
// ---------------------------------------------------------------------------
function showResults(result) {
    loadingState.style.display = "none";

    // Classification result
    const isOk = result.label === "ok";
    classLabel.textContent = result.label_fr;
    classLabel.className = `sim-class-label ${isOk ? "ok" : "def"}`;
    classConfidence.textContent = `Confiance: ${(result.confidence * 100).toFixed(1)}%`;
    classTime.textContent = `${result.inference_time_ms}ms`;
    classResult.style.display = "block";
    classResult.className = `sim-classification ${isOk ? "ok" : "def"}`;

    // Metric info
    metricInfo.textContent = `Métrique: ${result.metric || "cosine"}`;
    metricInfo.style.display = "block";

    // Animate classification in
    gsap.fromTo(classResult, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.3 });

    // Build carousel cards
    carousel.innerHTML = "";
    const similar = result.similar || [];

    if (similar.length === 0) {
        emptyState.style.display = "flex";
        emptyState.querySelector("p").textContent = "Aucune image similaire trouvée.";
        return;
    }

    similar.forEach((item, i) => {
        const card = document.createElement("div");
        card.className = "sim-card";
        const itemIsOk = item.label === "ok";

        card.innerHTML = `
            <div class="sim-card-rank">#${item.rank}</div>
            <div class="sim-card-img-wrapper">
                <img class="sim-card-img" src="${item.image_url}" alt="${item.path}" loading="lazy" />
            </div>
            <div class="sim-card-info">
                <span class="sim-card-label ${itemIsOk ? "ok" : "def"}">${itemIsOk ? "OK ✅" : "DEF ❌"}</span>
                <span class="sim-card-distance">dist: ${item.distance.toFixed(4)}</span>
            </div>
            <div class="sim-card-path">${item.path.split("/").pop()}</div>
        `;

        // Click to enlarge
        card.addEventListener("click", () => openOverlay(item));

        carousel.appendChild(card);

        // Stagger animation
        gsap.fromTo(card,
            { opacity: 0, y: 30, scale: 0.9 },
            { opacity: 1, y: 0, scale: 1, duration: 0.4, delay: i * 0.08, ease: "back.out(1.2)" }
        );
    });

    carouselWrapper.style.display = "flex";
}

// ---------------------------------------------------------------------------
// Carousel navigation
// ---------------------------------------------------------------------------
btnPrev.addEventListener("click", () => {
    carousel.scrollBy({ left: -220, behavior: "smooth" });
});
btnNext.addEventListener("click", () => {
    carousel.scrollBy({ left: 220, behavior: "smooth" });
});

// Keyboard navigation
document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") carousel.scrollBy({ left: -220, behavior: "smooth" });
    if (e.key === "ArrowRight") carousel.scrollBy({ left: 220, behavior: "smooth" });
    if (e.key === "Escape") closeOverlay();
});

// ---------------------------------------------------------------------------
// Image overlay (enlarged view)
// ---------------------------------------------------------------------------
function openOverlay(item) {
    overlayImage.src = item.image_url;
    const isOk = item.label === "ok";
    overlayInfo.innerHTML = `
        <strong>#${item.rank}</strong> —
        <span class="${isOk ? "text-green" : "text-red"}">${isOk ? "Conforme ✅" : "Défectueuse ❌"}</span> —
        Distance: ${item.distance.toFixed(6)} —
        <span class="text-muted">${item.path}</span>
    `;
    imageOverlay.style.display = "flex";
    gsap.fromTo(imageOverlay, { opacity: 0 }, { opacity: 1, duration: 0.2 });
    gsap.fromTo(".sim-overlay-content", { scale: 0.85 }, { scale: 1, duration: 0.3, ease: "back.out(1.4)" });
}

function closeOverlay() {
    gsap.to(imageOverlay, {
        opacity: 0,
        duration: 0.2,
        onComplete: () => { imageOverlay.style.display = "none"; },
    });
}

overlayClose.addEventListener("click", closeOverlay);
imageOverlay.addEventListener("click", (e) => {
    if (e.target === imageOverlay) closeOverlay();
});

// ---------------------------------------------------------------------------
// Global actions
// ---------------------------------------------------------------------------
function logout() {
    sessionStorage.clear();
    window.location.href = "/login.html";
}
