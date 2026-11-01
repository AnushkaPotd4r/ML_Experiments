/**
 * ML STUDIO - Client-Side Interactive Operations
 * Handles search filtering, category tabs, sidebar toggle,
 * AJAX predictions, and preset input populators.
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Sidebar Toggle for Mobile/Tablet
    const sidebarToggleBtn = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.sidebar');
    if (sidebarToggleBtn && sidebar) {
        sidebarToggleBtn.addEventListener('click', function () {
            sidebar.classList.toggle('show');
        });
    }

    // 2. Algorithm Dashboard Search and Category Filtering
    const searchInput = document.getElementById('searchAlgo');
    const filterTabs = document.querySelectorAll('.tab-btn');
    const algoCards = document.querySelectorAll('.algo-card');
    const totalVisibleCount = document.getElementById('visibleCount');

    let currentCategory = 'all';

    function filterAlgorithms() {
        if (!algoCards.length) return;
        const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
        let visible = 0;

        algoCards.forEach(card => {
            const name = card.getAttribute('data-name') ? card.getAttribute('data-name').toLowerCase() : '';
            const category = card.getAttribute('data-category') ? card.getAttribute('data-category').toLowerCase() : '';
            const desc = card.getAttribute('data-desc') ? card.getAttribute('data-desc').toLowerCase() : '';

            const matchesCategory = (currentCategory === 'all' || category === currentCategory);
            const matchesQuery = (!query || name.includes(query) || desc.includes(query) || category.includes(query));

            if (matchesCategory && matchesQuery) {
                card.style.display = 'flex';
                visible++;
            } else {
                card.style.display = 'none';
            }
        });

        if (totalVisibleCount) {
            totalVisibleCount.textContent = visible;
        }
    }

    if (searchInput) {
        searchInput.addEventListener('input', filterAlgorithms);
    }

    filterTabs.forEach(tab => {
        tab.addEventListener('click', function () {
            filterTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            currentCategory = this.getAttribute('data-filter').toLowerCase();
            filterAlgorithms();
        });
    });

    // 3. Preset Fill Buttons for Quick Testing
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', function () {
            const data = this.dataset;
            for (const key in data) {
                const input = document.getElementById(key);
                if (input) {
                    input.value = data[key];
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        });
    });
});
