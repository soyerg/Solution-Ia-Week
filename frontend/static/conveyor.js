/* =================================================================
   CONVEYOR.JS — GSAP-powered conveyor belt animation + API calls
   SPA-compatible: exposes window.Conveyor, uses AppHistory
   ================================================================= */

const Conveyor = (() => {
    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    const state = {
        queue: [],           // { file, objectUrl, name, status: 'waiting'|'processing'|'done' }
        isRunning: false,
        paused: false,       // true when similarity view is active
        stats: { total: 0, ok: 0, def: 0 },
        activeTl: null,      // current GSAP timeline (for pause/resume)
    };

    // -----------------------------------------------------------------------
    // DOM refs (scoped to #viewConveyor)
    // -----------------------------------------------------------------------
    const uploadZone    = document.getElementById("convUploadZone");
    const fileInput     = document.getElementById("convFileInput");
    const queueList     = document.getElementById("queueList");
    const queueCount    = document.getElementById("queueCount");
    const beltSurface   = document.getElementById("beltSurface");
    const beltChevrons  = document.getElementById("beltChevrons");
    const cameraLens    = document.getElementById("cameraLens");
    const cameraStatus  = document.getElementById("cameraStatus");
    const scanLine      = document.getElementById("scanLine");
    const resultDisplay = document.getElementById("resultDisplay");
    const resultContent = document.getElementById("resultContent");
    const binOk         = document.getElementById("binOk");
    const binDef        = document.getElementById("binDef");
    const binOkCount    = document.getElementById("binOkCount");
    const binDefCount   = document.getElementById("binDefCount");
    const statTotal     = document.getElementById("statTotal");
    const statOk        = document.getElementById("statOk");
    const statDef       = document.getElementById("statDef");
    const statRate      = document.getElementById("statRate");

    // -----------------------------------------------------------------------
    // Upload handling
    // -----------------------------------------------------------------------
    uploadZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        addFiles(e.target.files);
        fileInput.value = "";     // allow re-selecting same file
    });

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

    // -----------------------------------------------------------------------
    // Queue rendering
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Conveyor Loop — processes items sequentially
    // -----------------------------------------------------------------------
    async function runConveyor() {
        state.isRunning = true;

        while (true) {
            // Wait while paused
            while (state.paused) {
                await new Promise(r => setTimeout(r, 200));
            }

            const item = state.queue.find(q => q.status === "waiting");
            if (!item) break;

            item.status = "processing";
            renderQueue();

            await animatePieceCycle(item);

            item.status = "done";
            renderQueue();
        }

        state.isRunning = false;
    }

    // -----------------------------------------------------------------------
    // Animate one piece through the full conveyor cycle
    // -----------------------------------------------------------------------
    function animatePieceCycle(item) {
        return new Promise((resolve) => {
            const beltH = beltSurface.offsetHeight;
            const centerY = beltH / 2 - 40;

            const piece = document.createElement("div");
            piece.className = "piece";
            piece.innerHTML = `<img class="piece-img" src="${item.objectUrl}" alt="${item.name}" />`;
            beltSurface.appendChild(piece);

            const tl = gsap.timeline({
                onComplete: () => {
                    piece.remove();
                    state.activeTl = null;
                    resolve();
                },
            });

            state.activeTl = tl;

            // Phase 1: piece enters
            tl.to(piece, { top: centerY, duration: 1.5, ease: "power2.out" });

            // Phase 2: camera activates
            tl.call(() => {
                beltChevrons.classList.add("paused");
                cameraLens.classList.add("scanning");
                cameraStatus.textContent = "Classification...";
                cameraStatus.classList.add("classifying");
                scanLine.classList.add("active");
                updateResult("classifying", "🔄", "Analyse en cours...");
            });

            tl.to({}, { duration: 0.3 });

            // Phase 3: API call
            tl.addPause("api-call", async () => {
                let result;
                try {
                    result = await classifyImage(item.file);
                } catch (e) {
                    result = { label: "def", label_fr: "Erreur ❌", confidence: 0, inference_time_ms: 0 };
                }

                item.result = result;
                const isOk = result.label === "ok";

                scanLine.classList.remove("active");
                cameraLens.classList.remove("scanning");
                cameraStatus.textContent = result.label_fr;
                cameraStatus.classList.remove("classifying");

                updateResult(
                    isOk ? "ok" : "def",
                    isOk ? "✅" : "❌",
                    `${result.label_fr}  —  Confiance: ${(result.confidence * 100).toFixed(1)}%  —  ${result.inference_time_ms}ms`
                );

                piece.classList.add(isOk ? "result-ok" : "result-def");

                state.stats.total++;
                if (isOk) state.stats.ok++; else state.stats.def++;
                updateStats();

                // *** Add to shared history ***
                AppHistory.add(item.file, item.objectUrl, item.name, item.result);

                await new Promise(r => setTimeout(r, 800));

                // Sort animation
                beltChevrons.classList.remove("paused");
                cameraStatus.textContent = "En attente";

                const pieceRect = piece.getBoundingClientRect();
                const binOkRect = binOk.getBoundingClientRect();
                const binDefRect = binDef.getBoundingClientRect();

                const targetX = isOk
                    ? binOkRect.left + binOkRect.width / 2 - pieceRect.left - pieceRect.width / 2
                    : binDefRect.left + binDefRect.width / 2 - pieceRect.left - pieceRect.width / 2;

                gsap.to("#armPusher", {
                    x: isOk ? -30 : 30,
                    duration: 0.2,
                    yoyo: true,
                    repeat: 1,
                });

                await new Promise(resolveAnim => {
                    gsap.to(piece, {
                        x: targetX,
                        y: 40,
                        opacity: 0,
                        duration: 0.6,
                        ease: "power2.in",
                        onComplete: () => {
                            const bin = isOk ? binOk : binDef;
                            bin.classList.add(isOk ? "flash-ok" : "flash-def");
                            setTimeout(() => bin.classList.remove("flash-ok", "flash-def"), 600);

                            if (isOk) binOkCount.textContent = state.stats.ok;
                            else binDefCount.textContent = state.stats.def;
                            resolveAnim();
                        },
                    });
                });

                await new Promise(r => setTimeout(r, 200));
                tl.resume();
            });

            tl.to({}, { duration: 0.1 });
        });
    }

    // -----------------------------------------------------------------------
    // API call
    // -----------------------------------------------------------------------
    async function classifyImage(file) {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/api/classify", { method: "POST", body: formData });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    }

    // -----------------------------------------------------------------------
    // UI Helpers
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Pause / Resume (called by nav.js)
    // -----------------------------------------------------------------------
    function pause() {
        state.paused = true;
        // Pause active GSAP timeline if any
        if (state.activeTl) state.activeTl.pause();
        beltChevrons.classList.add("paused");
    }

    function resume() {
        state.paused = false;
        // Resume active GSAP timeline if any
        if (state.activeTl && state.activeTl.paused()) state.activeTl.resume();
        beltChevrons.classList.remove("paused");
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------
    return { pause, resume, state };
})();
