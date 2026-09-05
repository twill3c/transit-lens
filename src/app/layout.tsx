import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Transit Lens 観測卓 — transit-lens",
  description:
    "NASA Kepler が 4 年間測った恒星の明るさから、1 次元 CNN が惑星の通過を見つける。その判断を、専門知識なしで最後まで追える観測卓。推論はすべてブラウザの中で走る。",
};

// フリート共通のフッタ規約(koho-lens が正本)。
// MIT License ・ © ・ GitHub ・ 歩き方 ・ 設計図 ・ App Menu の 6 項目をこの並びで、
// position: fixed で常時表示する。**並びと項目数を揃えるのであって、文言は各アプリのものを残す。**
const FOOTER = {
  license: "https://github.com/twill3c/transit-lens/blob/main/LICENSE",
  repository: "https://github.com/twill3c/transit-lens",
  guide: "https://claude.ai/code/artifact/0dea8755-3f45-4b50-8015-ba47f48d25e6",
  blueprint: "https://claude.ai/code/artifact/BLUEPRINT_ID",
  appMenu: "https://app-menu-amber.vercel.app/",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Zen+Old+Mincho:wght@400;700&family=IBM+Plex+Sans+JP:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
        />
      </head>
      <body>
        {children}
        {/* fleet: fixed footer */}
        <footer className="site-footer">
          <div className="site-footer__inner">
            <a href={FOOTER.license}>MIT License</a>
            <span className="site-footer__copy">© 2026 坂田哲朗</span>
            <span className="fsep">・</span>
            <a href={FOOTER.repository}>GitHub</a>
            <span className="fsep">・</span>
            <a href={FOOTER.guide}>観測卓の歩き方</a>
            <span className="fsep">・</span>
            <a href={FOOTER.blueprint}>観測卓の設計図</a>
            <span className="fsep">・</span>
            <a href={FOOTER.appMenu}>App Menu</a>
          </div>
        </footer>
      </body>
    </html>
  );
}
