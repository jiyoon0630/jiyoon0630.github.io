# Protect inline math from kramdown.
#
# With `math_engine: null`, kramdown passes `$$...$$` through untouched but
# treats single-dollar `$...$` as ordinary Markdown text. Inside it, `\{` loses
# its backslash, `_..._` and `*...*` pairs turn into <em>, and `'` becomes a
# curly quote, so MathJax receives broken TeX. Rewriting each inline `$x$` as
# `$$x$$` before conversion makes kramdown emit it verbatim as `$x$` in a
# <span class="kdmath">, which MathJax then renders as inline math.
#
# Math is also written out raw, so a `<` inside it becomes `\lt`.
#
# Posts keep writing `$...$`. Skipped: posts with `math: false` (currency),
# code fences, inline code, and a line that is nothing but one `$x$` (kramdown
# would promote that to display math).
module ProtectInlineMath
  PATTERN = /
    (?<skip>
      ^(?<fence>```|~~~).*?^\k<fence>[^\n]*$   # code fence
    | `[^`\n]+`                                # inline code
    )
  | (?<display>\$\$.*?\$\$)                  # display math
  | (?<![\\$])\$(?<math>[^\s$](?:[^$\n]*?[^\s\\$])?)\$(?!\$)
  /mx

  def self.rewrite(text)
    text.gsub(PATTERN) do
      m = Regexp.last_match
      next m[0] if m[:skip]
      next lt(m[:display]) if m[:display]
      alone = m.pre_match.split("\n", -1).last.to_s.strip.empty? &&
              m.post_match.split("\n", -1).first.to_s.strip.empty?
      alone ? m[0] : "$$#{lt(m[:math])}$$"
    end
  end

  # kramdown writes math out verbatim, so `v_{<t}` reaches the browser as a
  # tag opening `<t`. `\lt` renders the same symbol.
  def self.lt(tex)
    tex.gsub(/<(?=\S)/, "\\lt ")
  end
end

Jekyll::Hooks.register :posts, :pre_render do |post|
  next if post.data["math"] == false
  post.content = ProtectInlineMath.rewrite(post.content)
end
