/* =================================================================
   SIMILARITY.JS — Upload + similarity search + carousel display
   SPA-compatible: exposes window.Similarity, uses AppHistory
   ================================================================= */

const Similarity = (() => {
    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    let currentFile = null;       // File object ready for /api/similar
    let currentBlobUrl = null;    // for preview display
    let searchDone = false;       // true once carousel is showing

    // -----------------------------------------------------------------------
    // DOM refs (scoped to #viewSimilarity)
    // -----------------------------------------------------------------------
    const queryPreview    = document.getElementById("queryPreview");
    const queryImage      = document.getElementById("queryImage");
    const queryInfo       = document.getElementById("queryInfo");
    const classResult     = document.getElementById("classificationResult");
    const classLabel      = document.getElementById("classLabel");
    const classConfidence = document.getElementById("classConfidence");
    const classTime       = document.getElementById("classTime");
    const metricInfo      = document.getElementById("metricInfo");
    const loadingState    = document.getElementById("loadingState");
    const emptyState      = document.getElementById("emptyState");
    const carouselWrapper = document.getElementById("carouselWrapper");
    const carousel        = document.getElementById("carousel");
    const btnPrev         = document.getElementById("btnPrev");
    const btnNext         = document.getElementById("btnNext");
    const imageOverlay    = document.getElementById("imageOverlay");
    const overlayImage    = document.getElementById("overlayImage");
    const overlayInfo     = document.getElementById("overlayInfo");
    const overlayClose    = document.getElementById("overlayClose");
    const btnSearch       = document.getElementById("btnSearchSimilar");
    const vllmDiagnosis   = document.getElementById("vllmDiagnosis");
    const vllmCategory    = document.getElementById("vllmCategory");
    const vllmDescription = document.getElementById("vllmDescription");

    // -----------------------------------------------------------------------
    // Load from history (called by nav.js via AppHistory.onSelect)
    // Pre-fills preview + classification info, shows search button
    // -----------------------------------------------------------------------
    function loadFromHistory(historyItem) {
        preloadPreview(historyItem.blobUrl, historyItem.name);
        currentFile = historyItem.file;
        currentBlobUrl = historyItem.blobUrl;
        searchDone = false;

        // Show classification result from history
        if (historyItem.result) {
            showClassification(historyItem.result);
        }

        // Show search button
        btnSearch.style.display = "block";

        // Reset carousel
        resetResults();
    }

    // -----------------------------------------------------------------------
    // Shared preview setup
    // -----------------------------------------------------------------------
    function preloadPreview(blobUrl, name) {
        queryImage.src = blobUrl;
        queryInfo.textContent = name;
        queryPreview.style.display = "block";

        gsap.fromTo(queryPreview,
            { opacity: 0, scale: 0.9 },
            { opacity: 1, scale: 1, duration: 0.4, ease: "back.out(1.4)" }
        );
    }

    function resetResults() {
        carouselWrapper.style.display = "none";
        emptyState.style.display = "flex";
        emptyState.querySelector("p").textContent =
                "Sélectionnez une image depuis l'historique puis lancez la recherche";
        metricInfo.style.display = "none";
        loadingState.style.display = "none";
        vllmDiagnosis.style.display = "none";
    }

    // -----------------------------------------------------------------------
    // Manual search trigger
    // -----------------------------------------------------------------------
    btnSearch.addEventListener("click", async () => {
        if (!currentFile) return;

        btnSearch.style.display = "none";
        emptyState.style.display = "none";
        carouselWrapper.style.display = "none";
        loadingState.style.display = "flex";

        try {
            const result = await searchSimilar(currentFile);
            showClassification(result);
            showResults(result);
            searchDone = true;

        } catch (e) {
            loadingState.style.display = "none";
            emptyState.style.display = "flex";
            emptyState.querySelector("p").textContent = `Erreur: ${e.message}`;
            btnSearch.style.display = "block";
        }
    });

    // -----------------------------------------------------------------------
    // API call
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Show classification info
    // -----------------------------------------------------------------------
    function showClassification(result) {
        const isOk = result.label === "ok";
        classLabel.textContent = result.label_fr;
        classLabel.className = `sim-class-label ${isOk ? "ok" : "def"}`;
        classConfidence.textContent = `Confiance: ${(result.confidence * 100).toFixed(1)}%`;
        classTime.textContent = `${result.inference_time_ms}ms`;
        classResult.style.display = "block";
        classResult.className = `sim-classification ${isOk ? "ok" : "def"}`;

        gsap.fromTo(classResult, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.3 });
    }

    // -----------------------------------------------------------------------
    // Display carousel results
    // -----------------------------------------------------------------------
    function showResults(result) {
        loadingState.style.display = "none";

        // Metric info
        metricInfo.textContent = `Métrique: ${result.metric || "cosine"}`;
        metricInfo.style.display = "block";

        // Build carousel cards
        carousel.innerHTML = "";
        const similar = result.similar || [];

        if (similar.length === 0 && !result.gradcam_overlay) {
            emptyState.style.display = "flex";
            emptyState.querySelector("p").textContent = "Aucune image similaire trouvée.";
            return;
        }

        // --- Grad-CAM card (first card, before similar images) ---
        let animIndex = 0;
        if (result.gradcam_overlay) {
            const gcCard = document.createElement("div");
            gcCard.className = "sim-card gradcam-card";
            const isOk = result.label === "ok";
            gcCard.innerHTML = `
                <div class="sim-card-rank gradcam-badge">🔍 Zones de décision</div>
                <div class="sim-card-img-wrapper">
                    <img class="sim-card-img" src="data:image/png;base64,${result.gradcam_overlay}" alt="Grad-CAM heatmap" />
                </div>
                <div class="sim-card-info">
                    <span class="sim-card-label ${isOk ? "ok" : "def"}">${isOk ? "OK ✅" : "DEF ❌"}</span>
                    <span class="sim-card-distance">Grad-CAM</span>
                </div>
                <div class="sim-card-path">${result.filename || "Image analysée"}</div>
            `;
            gcCard.addEventListener("click", () => {
                overlayImage.src = `data:image/png;base64,${result.gradcam_overlay}`;
                overlayInfo.innerHTML = `
                    <strong>🔍 Zones de décision</strong> —
                    <span class="${isOk ? 'text-green' : 'text-red'}">${isOk ? 'Conforme ✅' : 'Défectueuse ❌'}</span> —
                    Grad-CAM (Heatmap superposée sur l'image)
                `;
                imageOverlay.style.display = "flex";
                gsap.fromTo(imageOverlay, { opacity: 0 }, { opacity: 1, duration: 0.2 });
                gsap.fromTo(".sim-overlay-content", { scale: 0.85 }, { scale: 1, duration: 0.3, ease: "back.out(1.4)" });
            });
            carousel.appendChild(gcCard);
            gsap.fromTo(gcCard,
                { opacity: 0, y: 30, scale: 0.9 },
                { opacity: 1, y: 0, scale: 1, duration: 0.4, delay: 0, ease: "back.out(1.2)" }
            );
            animIndex = 1;
        }

        // --- Similar image cards ---
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

            card.addEventListener("click", () => openOverlay(item));
            carousel.appendChild(card);

            gsap.fromTo(card,
                { opacity: 0, y: 30, scale: 0.9 },
                { opacity: 1, y: 0, scale: 1, duration: 0.4, delay: (i + animIndex) * 0.08, ease: "back.out(1.2)" }
            );
        });

        carouselWrapper.style.display = "flex";

        // --- VLLM Diagnosis (only if piece is DEF and diagnosis available) ---
        showVllmDiagnosis(result);
    }

    // -----------------------------------------------------------------------
    // VLLM Diagnosis display
    // -----------------------------------------------------------------------
    function showVllmDiagnosis(result) {
        if (result.vllm_diagnosis && result.label === "def") {
            const diag = result.vllm_diagnosis;
            vllmCategory.textContent = diag.category;
            vllmDescription.textContent = diag.description;
            vllmDiagnosis.style.display = "flex";

            gsap.fromTo(vllmDiagnosis,
                { opacity: 0, y: 15 },
                { opacity: 1, y: 0, duration: 0.4, delay: 0.3, ease: "power2.out" }
            );
        } else {
            vllmDiagnosis.style.display = "none";
        }
    }

    // -----------------------------------------------------------------------
    // Carousel navigation
    // -----------------------------------------------------------------------
    btnPrev.addEventListener("click", () => {
        carousel.scrollBy({ left: -220, behavior: "smooth" });
    });
    btnNext.addEventListener("click", () => {
        carousel.scrollBy({ left: 220, behavior: "smooth" });
    });

    document.addEventListener("keydown", (e) => {
        // Only handle if similarity view is visible
        const simView = document.getElementById("viewSimilarity");
        if (simView.style.display === "none") return;

        if (e.key === "ArrowLeft") carousel.scrollBy({ left: -220, behavior: "smooth" });
        if (e.key === "ArrowRight") carousel.scrollBy({ left: 220, behavior: "smooth" });
        if (e.key === "Escape") closeOverlay();
    });

    // -----------------------------------------------------------------------
    // Image overlay
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------
    return { loadFromHistory };
})();
