/* =================================================================
   CONVEYOR.JS — GSAP-powered conveyor belt animation + API calls
   ================================================================= */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
    queue: [],           // { file, objectUrl, name, status: 'waiting'|'processing'|'done' }
    isRunning: false,
    stats: { total: 0, ok: 0, def: 0 },
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const uploadZone    = $("#uploadZone");
const fileInput     = $("#fileInput");
const queueList     = $("#queueList");
const queueCount    = $("#queueCount");
const beltSurface   = $("#beltSurface");
const beltChevrons  = $("#beltChevrons");
const cameraLens    = $("#cameraLens");
const cameraStatus  = $("#cameraStatus");
const scanLine      = $("#scanLine");
const resultDisplay = $("#resultDisplay");
const resultContent = $("#resultContent");
const historyList   = $("#historyList");
const binOk         = $("#binOk");
const binDef        = $("#binDef");
const binOkCount    = $("#binOkCount");
const binDefCount   = $("#binDefCount");
const statTotal     = $("#statTotal");
const statOk        = $("#statOk");
const statDef       = $("#statDef");
const statRate      = $("#statRate");

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
fileInput.addEventListener("change", (e) => addFiles(e.target.files));

// Drag & Drop
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    addFiles(e.dataTransfer.files);
});

function addFiles(files) {
    for (const file of files) {
        if (!file.type.startsWith("image/")) continue;
        state.queue.push({
            file,
            objectUrl: URL.createObjectURL(file),
            name: file.name,
            status: "waiting",
        });
    }
    renderQueue();
    if (!state.isRunning) runConveyor();
}

// ---------------------------------------------------------------------------
// Queue rendering
// ---------------------------------------------------------------------------
function renderQueue() {
    queueCount.textContent = state.queue.filter(q => q.status === "waiting").length;
    queueList.innerHTML = "";
    for (const item of state.queue) {
        const el = document.createElement("div");
        el.className = `queue-item ${item.status === "processing" ? "active" : ""} ${item.status === "done" ? "done" : ""}`;
        el.innerHTML = `
            <img class="queue-item-thumb" src="${item.objectUrl}" alt="" />
            <span class="queue-item-name">${item.name}</span>
            <span class="queue-item-status">${
                item.status === "waiting"    ? "⏳" :
                item.status === "processing" ? "🔄" : "✅"
            }</span>
        `;
        queueList.appendChild(el);
    }
}

// ---------------------------------------------------------------------------
// Main Conveyor Loop — processes items sequentially
// ---------------------------------------------------------------------------
async function runConveyor() {
    state.isRunning = true;

    while (true) {
        // Find next waiting item
        const item = state.queue.find(q => q.status === "waiting");
        if (!item) break;

        item.status = "processing";
        renderQueue();

        // Animate one full cycle for this piece
        await animatePieceCycle(item);

        item.status = "done";
        renderQueue();
    }

    state.isRunning = false;
}

// ---------------------------------------------------------------------------
// Animate one piece through the full conveyor cycle
// ---------------------------------------------------------------------------
function animatePieceCycle(item) {
    return new Promise((resolve) => {
        const beltRect = beltSurface.getBoundingClientRect();
        const beltH = beltSurface.offsetHeight;
        const centerY = beltH / 2 - 40; // center piece on camera

        // Create piece element
        const piece = document.createElement("div");
        piece.className = "piece";
        piece.innerHTML = `<img class="piece-img" src="${item.objectUrl}" alt="${item.name}" />`;
        beltSurface.appendChild(piece);

        // GSAP Master timeline
        const tl = gsap.timeline({
            onComplete: () => {
                piece.remove();
                resolve();
            },
        });

        // === Phase 1: Belt moves, piece enters from top ===
        tl.to(piece, {
            top: centerY,
            duration: 1.5,
            ease: "power2.out",
        });

        // === Phase 2: Belt stops, camera activates ===
        tl.call(() => {
            beltChevrons.classList.add("paused");
            cameraLens.classList.add("scanning");
            cameraStatus.textContent = "Classification...";
            cameraStatus.classList.add("classifying");
            scanLine.classList.add("active");
            updateResult("classifying", "🔄", "Analyse en cours...");
        });

        // Small pause for visual effect before API call
        tl.to({}, { duration: 0.3 });

        // === Phase 3: API call (timeline pauses here) ===
        tl.addPause("api-call", async () => {
            let result;
            try {
                result = await classifyImage(item.file);
            } catch (e) {
                result = { label: "def", label_fr: "Erreur ❌", confidence: 0, inference_time_ms: 0 };
            }

            // Store result on item
            item.result = result;

            // Update camera UI
            scanLine.classList.remove("active");
            cameraLens.classList.remove("scanning");
            cameraStatus.textContent = result.label_fr;
            cameraStatus.classList.remove("classifying");

            // Update result display
            const isOk = result.label === "ok";
            updateResult(
                isOk ? "ok" : "def",
                isOk ? "✅" : "❌",
                `${result.label_fr}  —  Confiance: ${(result.confidence * 100).toFixed(1)}%  —  ${result.inference_time_ms}ms`
            );

            // Highlight piece
            piece.classList.add(isOk ? "result-ok" : "result-def");

            // Update stats
            state.stats.total++;
            if (isOk) state.stats.ok++; else state.stats.def++;
            updateStats();

            // Add to history
            addHistoryItem(item);

            // Resume timeline after a brief display pause
            setTimeout(() => tl.resume(), 800);
        });

        // === Phase 4: Belt resumes, piece gets sorted ===
        tl.call(() => {
            beltChevrons.classList.remove("paused");
            cameraStatus.textContent = "En attente";
        });

        // Decide sort direction based on result
        tl.call(() => {
            const isOk = item.result && item.result.label === "ok";
            const exitX = isOk ? -200 : 200;  // left = OK, right = Defect
            const exitY = centerY + 40;

            // Animate sorting arm
            gsap.to("#armPusher", {
                x: isOk ? -30 : 30,
                duration: 0.2,
                yoyo: true,
                repeat: 1,
            });

            // Animate piece to bin
            gsap.to(piece, {
                x: exitX,
                top: exitY + 20,
                opacity: 0,
                duration: 0.6,
                ease: "power2.in",
                onComplete: () => {
                    // Flash bin
                    const bin = isOk ? binOk : binDef;
                    bin.classList.add(isOk ? "flash-ok" : "flash-def");
                    setTimeout(() => bin.classList.remove("flash-ok", "flash-def"), 600);

                    // Update bin count
                    if (isOk) {
                        binOkCount.textContent = state.stats.ok;
                    } else {
                        binDefCount.textContent = state.stats.def;
                    }
                },
            });
        });

        // Wait for sort animation to complete
        tl.to({}, { duration: 1 });
    });
}

// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------
async function classifyImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/classify", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    return await response.json();
}

// ---------------------------------------------------------------------------
// UI Helpers
// ---------------------------------------------------------------------------
function updateResult(type, icon, text) {
    resultDisplay.className = `result-display ${type === "classifying" ? "" : type}`;
    resultContent.innerHTML = `
        <span class="result-icon">${icon}</span>
        <span class="result-text">${text}</span>
    `;
}

function updateStats() {
    statTotal.textContent = state.stats.total;
    statOk.textContent = state.stats.ok;
    statDef.textContent = state.stats.def;
    statRate.textContent = state.stats.total > 0
        ? `${((state.stats.ok / state.stats.total) * 100).toFixed(1)}%`
        : "—";
}

function addHistoryItem(item) {
    // Remove empty message
    const empty = historyList.querySelector(".history-empty");
    if (empty) empty.remove();

    const isOk = item.result.label === "ok";
    const el = document.createElement("div");
    el.className = "history-item";
    el.innerHTML = `
        <img class="history-thumb" src="${item.objectUrl}" alt="" />
        <div class="history-info">
            <div class="history-name">${item.name}</div>
            <div class="history-result ${isOk ? "ok" : "def"}">${item.result.label_fr}</div>
        </div>
        <div class="history-time">${item.result.inference_time_ms}ms</div>
    `;

    // Insert at top
    historyList.insertBefore(el, historyList.firstChild);
}

// ---------------------------------------------------------------------------
// Global actions
// ---------------------------------------------------------------------------
function clearHistory() {
    historyList.innerHTML = '<div class="history-empty">Aucune pièce analysée</div>';
}

function logout() {
    sessionStorage.clear();
    window.location.href = "/login.html";
}
