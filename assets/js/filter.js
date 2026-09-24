// Note pages: mark tables whose content is wider than the text column, so the
// CSS widens only those into the margins (see .paper-body table.wide).
document.addEventListener('DOMContentLoaded', function () {
  var body = document.querySelector('.paper-body');
  if (!body) return;
  var tables = body.querySelectorAll('table');
  if (!tables.length) return;

  function hPad(el) {
    var cs = getComputedStyle(el);
    return parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight) +
           parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth);
  }

  function mark() {
    var column = parseFloat(getComputedStyle(body).width);
    tables.forEach(function (t) {
      // A callout's own padding narrows the room its table has.
      var quote = t.closest('blockquote');
      var room = column - (quote ? hPad(quote) : 0);
      var old = t.style.width;
      t.style.width = 'max-content';
      var natural = t.scrollWidth;
      t.style.width = old;
      var wide = natural > room + 1;
      t.classList.toggle('wide', wide);
      // The CSS grows the table (or its callout) only as far as this.
      var target = quote || t;
      if (wide) {
        target.style.setProperty('--natural',
          Math.ceil(natural + (quote ? hPad(quote) : 0)) + 'px');
      } else {
        target.style.removeProperty('--natural');
      }
    });
  }

  mark();
  // Web fonts change text widths once they arrive.
  window.addEventListener('load', mark);
  var timer;
  window.addEventListener('resize', function () {
    clearTimeout(timer);
    timer = setTimeout(mark, 150);
  });
});

// Notes page: combined kind + language + tag filtering.
document.addEventListener('DOMContentLoaded', function () {
  var kindButtons = document.querySelectorAll('.kind-btn');
  var langButtons = document.querySelectorAll('.lang-btn');
  var tagButtons = document.querySelectorAll('.filter-btn');
  var items = document.querySelectorAll('.paper-item');
  var empty = document.getElementById('empty-lang');
  // Only filter on the Notes page, where the controls exist. Elsewhere
  // (e.g. the home page list) items must stay visible.
  if (!items.length || !document.getElementById('tag-filters')) return;

  var curKind = 'all';
  var curLang = 'en'; // default: English-first
  var curTag = 'all';

  function setActive(buttons, btn) {
    buttons.forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
  }

  // Each kind tab counts notes in the selected language only.
  function updateCounts() {
    kindButtons.forEach(function (btn) {
      var span = btn.querySelector('.kind-count');
      if (!span) return;
      var kind = btn.getAttribute('data-kind');
      var n = 0;
      items.forEach(function (item) {
        if (item.getAttribute('data-lang') === curLang &&
            item.getAttribute('data-kind') === kind) n++;
      });
      span.textContent = n;
    });
  }

  function apply() {
    updateCounts();
    var visible = 0;
    items.forEach(function (item) {
      var lang = item.getAttribute('data-lang');
      var kind = item.getAttribute('data-kind');
      var tags = (item.getAttribute('data-tags') || '').trim().split(/\s+/);
      var show = lang === curLang &&
                 (curKind === 'all' || kind === curKind) &&
                 (curTag === 'all' || tags.indexOf(curTag) !== -1);
      item.style.display = show ? '' : 'none';
      if (show) visible++;
    });
    if (empty) empty.hidden = visible !== 0;
  }

  function wire(buttons, attr, set) {
    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        set(btn.getAttribute(attr));
        setActive(buttons, btn);
        apply();
      });
    });
  }

  wire(kindButtons, 'data-kind', function (v) { curKind = v; });
  wire(langButtons, 'data-lang', function (v) { curLang = v; });
  wire(tagButtons, 'data-tag', function (v) { curTag = v; });

  apply();
});
