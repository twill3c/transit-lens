import { Observatory } from "@/components/Observatory";
import { DrawPad } from "@/components/DrawPad";
import { Sandbox } from "@/components/Sandbox";
import { Scorecard } from "@/components/Scorecard";
import { ShapePair } from "@/components/ShapePair";
import { Threshold } from "@/components/Threshold";

export default function Page() {
  return (
    <div className="wrap">
      <header className="masthead">
        <div>
          <p className="eyebrow">NASA Kepler × 1 次元 CNN</p>
          <h1>
            <span className="en">Transit Lens</span>
            <br />
            観測卓
          </h1>
          <p className="lede">
            恒星の明るさのわずかな揺らぎから、ニューラルネットワークが惑星を見つける。
            その判断を、専門知識なしで最後まで追えるようにした画面。入力は三通り、出力は五層。
            推論はすべてこのブラウザの中で走る。
          </p>
        </div>
        <span className="badge">
          NASA Kepler の<b>実データ</b>／ブラウザ内推論
        </span>
      </header>

      <Observatory />
      <Sandbox />
      <DrawPad />
      <ShapePair />
      <Threshold />

      <section className="step">
        <div className="step-head">
          <span className="step-n">答え合わせ</span>
          <h2>この模型はどこまで本物か</h2>
          <span className="io">OUT: 実測値だけ</span>
        </div>
        <p className="note">
          「よく当たる」と言うだけなら誰にでもできる。ここに出る数はすべて、
          <b>学習にも検証にも使っていない星</b>で測ったものである。
          比較先は Shallue &amp; Vanderburg (2018) の AstroNet ——
          同じラベル集合(Q1–Q17 DR24 の教師つき TCE 15,737 件)を使っているので、
          数字を並べて読める。
        </p>
        <Scorecard />
        <p className="foot">
          <b>陰性対照の物差しは、作りながら二度書き直した。</b>
          最初は「AUC が 0.55 未満なら合格」と決めていたが、
          <b>未学習</b>のモデル(学習を一度もしていない)の AUC を初期値 5 通りで測ると
          0.3512〜0.7245 に散った —— <b>乱数だけで 0.72 が出る</b>ので、
          0.5 との近さでは判定できない。
          次に実測すると、ラベルを壊して学習したモデルの出力は
          <b>ほぼ定数</b>(標準偏差 0.00177)になり、その微小な揺らぎが信号の強さと
          順位相関 +0.60 を持つため、<b>AUC は 0.117 まで下がった</b> ——
          何も学んでいないのに 0.5 から大きく離れる。
          だから順位ではなく<b>クラスの離れ具合</b>で見ている。
          <br />
          頭部の比較も同じ 22 epoch で回した。published と同じ全結合型は 0.9682、
          いま出荷している CAM 型は 0.9679。<b>差は 0.0003 で、
          視線が厳密になることの代償は AUC にはほとんど無かった</b>
          —— ただし収束は全結合型の方が速く、こちらが 22 epoch かけて届く水準に
          4 epoch で達する。パラメータ数は 31 倍違う。
        </p>
      </section>

      <section className="rules">
        <p className="eyebrow">この画面で守る取り決め</p>
        <h2>わかりやすさは、削ることではなく順番</h2>
        <ul>
          <li>
            <span className="rn">—</span>
            <span>
              <b>絵が先、数字が後。</b>まず星と惑星が動く。ppm も AUC も、その下に置く。
              順番を変えるだけで、専門語を消さずに済む。
            </span>
          </li>
          <li>
            <span className="rn">—</span>
            <span>
              <b>専門語には必ず言い換えを添える。</b>「深さ 158 ppm」の隣に
              「明るさが 0.0158% 下がる」。用語を消すと学べなくなるので、消さずに翻訳する。
            </span>
          </li>
          <li>
            <span className="rn">—</span>
            <span>
              <b>誇張したら明記する。</b>惑星の見かけの大きさは実寸ではない。
              地球サイズの窪みは画面上で 0.5 ピクセルにもならないので、
              模式図と断ったうえで拡大する。視線を曲線の長さへ引き伸ばしていることも同じく書く。
            </span>
          </li>
          <li>
            <span className="rn">—</span>
            <span>
              <b>正解を見せる前に、判定を見せる。</b>先に答えを出すと、モデルの言い分を読まなくなる。
              答え合わせは必ず一段下に置く。
            </span>
          </li>
          <li>
            <span className="rn">—</span>
            <span>
              <b>失敗も出力の一部として扱う。</b>誤検出の例を最初のギャラリーに混ぜてある。
              全部当たる画面は、何も教えない。
            </span>
          </li>
          <li>
            <span className="rn">—</span>
            <span>
              <b>合成と実測を混ぜない。</b>ギャラリーと成績は Kepler の実データ。
              つまみの曲線は合成だが、<b>同じ前処理と同じモデル</b>を通している。
              どちらがどちらかは、その場に書いてある。
            </span>
          </li>
        </ul>
        <p className="foot">
          データ出典：NASA Exoplanet Archive（Q1–Q17 DR24 TCE 台帳・
          <code>av_training_set</code> を教師ラベルとして使用）、MAST（Kepler 長時間ケイデンス
          光度曲線 PDCSAP）。いずれも NASA 制作物でパブリックドメイン。
          前処理は Shallue &amp; Vanderburg (2018) <i>AJ</i> 155:94 の §3 に合わせている。
          <br />
          本アプリは既知の TCE に対する再現であり、新しい惑星の発見を主張するものではない。
        </p>
      </section>
    </div>
  );
}
