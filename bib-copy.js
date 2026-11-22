// Adds a "copy" button to the top-right of every .bib box.
// The copied text is rebuilt from the rows so it is properly formatted BibTeX:
//   @inproceedings{key,
//     author = {...},
//   }
(function () {
    const COPY_ICON =
        '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    const CHECK_ICON =
        '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<polyline points="20 6 9 17 4 12"/></svg>';

    function bibText(bib) {
        return Array.from(bib.querySelectorAll('.row')).map(row => {
            const cells = row.querySelectorAll('span');
            if (cells.length >= 3) {
                return '  ' + cells[0].textContent.trim() + ' = ' + cells[2].textContent.trim();
            }
            return row.textContent.trim();
        }).join('\n');
    }

    async function copyText(text) {
        try {
            await navigator.clipboard.writeText(text);
        } catch (e) {
            // Fallback for older browsers / non-secure contexts (e.g. file://)
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            ta.remove();
        }
    }

    document.querySelectorAll('.bib').forEach(bib => {
        // Wrap the box so the button stays put when the box scrolls horizontally.
        const wrap = document.createElement('div');
        wrap.className = 'bib-wrap';
        bib.parentNode.insertBefore(wrap, bib);
        wrap.appendChild(bib);

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'bib-copy';
        btn.title = 'Copy BibTeX';
        btn.setAttribute('aria-label', 'Copy BibTeX');
        btn.innerHTML = COPY_ICON;
        wrap.appendChild(btn);

        let timer;
        btn.addEventListener('click', async () => {
            await copyText(bibText(bib));
            btn.innerHTML = CHECK_ICON;
            btn.classList.add('copied');
            btn.title = 'Copied!';
            clearTimeout(timer);
            timer = setTimeout(() => {
                btn.innerHTML = COPY_ICON;
                btn.classList.remove('copied');
                btn.title = 'Copy BibTeX';
            }, 1500);
        });
    });
})();
