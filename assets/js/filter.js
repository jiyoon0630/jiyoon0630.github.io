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

  function apply() {
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
