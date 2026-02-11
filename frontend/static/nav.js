/* =================================================================
   NAV.JS — SPA view switching + shared history wiring
   ================================================================= */

(() => {
    // ---------- DOM refs ----------
    const viewConveyor    = document.getElementById("viewConveyor");
    const viewSimilarity  = document.getElementById("viewSimilarity");
    const navConveyor     = document.getElementById("navConveyor");
    const navSimilarity   = document.getElementById("navSimilarity");
    const headerIcon      = document.getElementById("headerIcon");
    const headerTitle     = document.getElementById("headerTitle");
    const btnClear        = document.getElementById("btnClearHistory");

    let currentView = "conveyor";

    // ---------- View switching ----------

    function switchView(view) {
        if (view === currentView) return;
        currentView = view;

        if (view === "conveyor") {
            viewConveyor.style.display = "";
            viewSimilarity.style.display = "none";
            navConveyor.classList.add("active");
            navSimilarity.classList.remove("active");
            headerIcon.textContent = "🏭";
            headerTitle.textContent = "Contrôle Qualité — Convoyeur IA";

            // Resume conveyor
            if (typeof Conveyor !== "undefined") Conveyor.resume();
        } else {
            viewConveyor.style.display = "none";
            viewSimilarity.style.display = "";
            navConveyor.classList.remove("active");
            navSimilarity.classList.add("active");
            headerIcon.textContent = "🔍";
            headerTitle.textContent = "Recherche de Similarité — IA";

            // Pause conveyor
            if (typeof Conveyor !== "undefined") Conveyor.pause();
        }
    }

    navConveyor.addEventListener("click", () => switchView("conveyor"));
    navSimilarity.addEventListener("click", () => switchView("similarity"));

    // ---------- History → Similarity bridge ----------

    AppHistory.setOnSelect((item) => {
        // Switch to similarity view
        switchView("similarity");

        // Pre-load the image from history (don't auto-search)
        if (typeof Similarity !== "undefined") {
            Similarity.loadFromHistory(item);
        }
    });

    // ---------- Clear history ----------
    btnClear.addEventListener("click", () => {
        AppHistory.clear();
    });

    // ---------- Backend health check (single instance) ----------
    async function checkBackend() {
        const dot  = document.querySelector(".status-dot");
        const text = document.querySelector(".status-text");
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

    // ---------- Global logout ----------
    window.logout = function () {
        sessionStorage.clear();
        window.location.href = "/login.html";
    };

    // Expose switchView globally for programmatic use
    window.switchView = switchView;
})();
