// Dark mode: remember the choice, otherwise follow the system setting
(function () {
    const root = document.documentElement;
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = saved || (prefersDark ? 'dark' : 'light');

    document.addEventListener('DOMContentLoaded', () => {
        const toggle = document.getElementById('theme-toggle');
        const update = () => { toggle.textContent = root.dataset.theme === 'dark' ? '☀️' : '🌙'; };
        update();
        toggle.addEventListener('click', () => {
            root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('theme', root.dataset.theme);
            update();
        });

        const language = document.getElementById('language');
        language.addEventListener('change', () => { window.location.href = '/language/' + language.value; });
    });
})();
