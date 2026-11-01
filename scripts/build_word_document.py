"""
Builds TradeSwarm_Architecture_and_Agents_Guide.docx natively using OpenXML.
No external dependencies required (uses standard library zipfile and xml).
"""
import zipfile
import html
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE_1 = PROJECT_ROOT / "TradeSwarm_Architecture_and_Agents_Guide.docx"
OUTPUT_FILE_2 = PROJECT_ROOT / "docs" / "TradeSwarm_Architecture_and_Agents_Guide.docx"

CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

WORD_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:color w:val="262626"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="160" w:line="260" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>

  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="360" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="50"/>
      <w:color w:val="0F355C"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="0" w:after="300"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:i/>
      <w:sz w:val="26"/>
      <w:color w:val="595959"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="320" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/>
      <w:b/>
      <w:sz w:val="34"/>
      <w:color w:val="0F355C"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="240" w:after="80"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/>
      <w:b/>
      <w:sz w:val="28"/>
      <w:color w:val="1F628D"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="180" w:after="60"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/>
      <w:b/>
      <w:sz w:val="24"/>
      <w:color w:val="2E75B6"/>
    </w:rPr>
  </w:style>
</w:styles>"""


class DocxBuilder:
    def __init__(self):
        self.body_xml = []

    def escape(self, text: str) -> str:
        return html.escape(str(text))

    def add_title(self, text: str):
        self.body_xml.append(
            f'<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>'
            f'<w:r><w:t>{self.escape(text)}</w:t></w:r></w:p>'
        )

    def add_subtitle(self, text: str):
        self.body_xml.append(
            f'<w:p><w:pPr><w:pStyle w:val="Subtitle"/></w:pPr>'
            f'<w:r><w:t>{self.escape(text)}</w:t></w:r></w:p>'
        )

    def add_h1(self, text: str):
        self.body_xml.append(
            f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
            f'<w:r><w:t>{self.escape(text)}</w:t></w:r></w:p>'
        )

    def add_h2(self, text: str):
        self.body_xml.append(
            f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>'
            f'<w:r><w:t>{self.escape(text)}</w:t></w:r></w:p>'
        )

    def add_h3(self, text: str):
        self.body_xml.append(
            f'<w:p><w:pPr><w:pStyle w:val="Heading3"/></w:pPr>'
            f'<w:r><w:t>{self.escape(text)}</w:t></w:r></w:p>'
        )

    def add_p(self, text: str, bold_prefix: str = ""):
        runs = []
        if bold_prefix:
            runs.append(f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{self.escape(bold_prefix)} </w:t></w:r>')
        runs.append(f'<w:r><w:t>{self.escape(text)}</w:t></w:r>')
        self.body_xml.append(f'<w:p>{"".join(runs)}</w:p>')

    def add_bullet(self, bold_prefix: str, text: str):
        self.body_xml.append(
            f'<w:p><w:pPr><w:ind w:left="480" w:hanging="240"/><w:spacing w:after="100"/></w:pPr>'
            f'<w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:color w:val="0F355C"/><w:b/></w:rPr><w:t xml:space="preserve">▪  </w:t></w:r>'
            f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{self.escape(bold_prefix)} </w:t></w:r>'
            f'<w:r><w:t>{self.escape(text)}</w:t></w:r></w:p>'
        )

    def add_callout(self, text: str, title: str = "KEY TAKEAWAY"):
        self.body_xml.append(
            f'<w:p><w:pPr>'
            f'<w:pBdr>'
            f'  <w:left w:val="single" w:sz="36" w:space="16" w:color="0F355C"/>'
            f'  <w:top w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'  <w:bottom w:val="none"/>'
            f'</w:pBdr>'
            f'<w:shd w:val="clear" w:color="auto" w:fill="F0F4F8"/>'
            f'<w:spacing w:before="180" w:after="180"/>'
            f'<w:ind w:left="240" w:right="240"/>'
            f'</w:pPr>'
            f'<w:r><w:rPr><w:b/><w:color w:val="0F355C"/></w:rPr><w:t xml:space="preserve">[{self.escape(title)}] </w:t></w:r>'
            f'<w:r><w:t>{self.escape(text)}</w:t></w:r>'
            f'</w:p>'
        )

    def add_code_block(self, text: str):
        lines = text.strip().split("\n")
        para_elements = []
        for i, line in enumerate(lines):
            before = "100" if i == 0 else "0"
            after = "100" if i == len(lines) - 1 else "0"
            para_elements.append(
                f'<w:p><w:pPr>'
                f'<w:shd w:val="clear" w:color="auto" w:fill="F4F6F9"/>'
                f'<w:spacing w:before="{before}" w:after="{after}" w:line="220" w:lineRule="auto"/>'
                f'<w:ind w:left="360" w:right="360"/>'
                f'</w:pPr>'
                f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="19"/><w:color w:val="1E293B"/></w:rPr>'
                f'<w:t xml:space="preserve">{self.escape(line)}</w:t></w:r></w:p>'
            )
        self.body_xml.extend(para_elements)

    def add_table(self, headers: list, rows: list, col_widths: list = None):
        col_w = col_widths or [2500] * len(headers)
        tbl_w = sum(col_w)
        
        xml = [
            f'<w:tbl><w:tblPr>'
            f'<w:tblW w:w="{tbl_w}" w:type="dxa"/>'
            f'<w:tblBorders>'
            f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="0F355C"/>'
            f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="0F355C"/>'
            f'  <w:left w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D9E1E8"/>'
            f'  <w:insideV w:val="none"/>'
            f'</w:tblBorders>'
            f'<w:tblCellMar>'
            f'  <w:top w:w="120" w:type="dxa"/><w:bottom w:w="120" w:type="dxa"/>'
            f'  <w:left w:w="160" w:type="dxa"/><w:right w:w="160" w:type="dxa"/>'
            f'</w:tblCellMar>'
            f'</w:tblPr>'
        ]

        # Header Row
        xml.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>')
        for idx, h in enumerate(headers):
            w = col_w[idx] if idx < len(col_w) else 2500
            xml.append(
                f'<w:tc><w:tcPr>'
                f'<w:tcW w:w="{w}" w:type="dxa"/>'
                f'<w:shd w:val="clear" w:color="auto" w:fill="0F355C"/>'
                f'</w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:after="60"/></w:pPr>'
                f'<w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr>'
                f'<w:t>{self.escape(h)}</w:t></w:r></w:p>'
                f'</w:tc>'
            )
        xml.append('</w:tr>')

        # Body Rows
        for r_idx, row in enumerate(rows):
            bg = "F9FBFC" if r_idx % 2 == 1 else "FFFFFF"
            xml.append('<w:tr>')
            for idx, cell in enumerate(row):
                w = col_w[idx] if idx < len(col_w) else 2500
                xml.append(
                    f'<w:tc><w:tcPr>'
                    f'<w:tcW w:w="{w}" w:type="dxa"/>'
                    f'<w:shd w:val="clear" w:color="auto" w:fill="{bg}"/>'
                    f'</w:tcPr>'
                    f'<w:p><w:pPr><w:spacing w:after="40"/></w:pPr>'
                    f'<w:r><w:t>{self.escape(str(cell))}</w:t></w:r></w:p>'
                    f'</w:tc>'
                )
            xml.append('</w:tr>')

        xml.append('</w:tbl>')
        self.body_xml.append("".join(xml))
        # Add space after table
        self.body_xml.append('<w:p><w:pPr><w:spacing w:after="160"/></w:pPr></w:p>')

    def add_page_break(self):
        self.body_xml.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def build_docx(self, output_path: Path):
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
            '<w:body>\n' +
            "\n".join(self.body_xml) +
            '\n<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>\n'
            '</w:body>\n</w:document>'
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
            zf.writestr("_rels/.rels", ROOT_RELS_XML)
            zf.writestr("word/_rels/document.xml.rels", WORD_RELS_XML)
            zf.writestr("word/styles.xml", STYLES_XML)
            zf.writestr("word/document.xml", doc_xml)


def generate_full_guide():
    b = DocxBuilder()

    # Document Header / Title
    b.add_title("TradeSwarm: Multi-Agent Trading Architecture")
    b.add_subtitle("Comprehensive System Specification, Agent Deliberation Mechanics, and Mathematical Formulation")
    
    b.add_callout(
        "TradeSwarm is a multi-agent reinforcement learning and algorithmic trading architecture specifically engineered "
        "to survive coordinated cryptocurrency manipulation ('Pump and Dump' schemes). Its primary novelty is an "
        "Independent Sentiment-Anomaly Veto Agent with non-negotiable executive override authority, decoupled from profit incentives.",
        "PROJECT EXECUTIVE SUMMARY"
    )

    # SECTION 1: HOW THE PROJECT WORKS
    b.add_h1("1. How the TradeSwarm System Works")
    
    b.add_h2("1.1 The Fundamental Problem: Coordinated Market Manipulation")
    b.add_p(
        "In decentralized cryptocurrency markets (e.g., Binance Spot markets pairing altcoins with BTC), manipulative "
        "syndicates coordinate via encrypted channels (such as Telegram). A pump-and-dump event unfolds in three tactical stages:"
    )
    b.add_bullet("Phase A: Silent Accumulation", "Organizers quietly accumulate a micro-cap cryptocurrency over several hours or days without triggering volume alerts.")
    b.add_bullet("Phase B: The Coordinated Surge", "At a pre-announced second, organizers broadcast aggressive buy calls ('TARGET 300%!', 'PUMP STARTED!', 'MOON!') across channels with hundreds of thousands of followers.")
    b.add_bullet("Phase C: The Instant Dump", "Within 15 to 90 seconds, organizers dump their pre-accumulated inventory into the incoming retail bids. The price collapses violently, leaving late buyers with an average instantaneous loss of 40.72%.")

    b.add_h2("1.2 The Fatal Flaw in Existing Multi-Agent Bots (Singh, 2026)")
    b.add_p(
        "Prior academic state-of-the-art architectures (such as Khushvir Singh, 2026) combine technical market indicators, "
        "sentiment analysis, and LSTM price predictions into a democratic majority vote or joint reinforcement learning state vector. "
        "When a pump erupts, all three conventional agents get tricked simultaneously:"
    )
    b.add_bullet("Market Agent:", "Sees the violent green candle and surging volume -> Votes BUY (1).")
    b.add_bullet("Sentiment Agent (FinBERT):", "Scoring algorithms read 'BUY NOW! HUGE PUMP TO THE MOON!' as overwhelmingly bullish -> Votes BUY (1).")
    b.add_bullet("Prediction Agent (LSTM):", "Extrapolates recent upward velocity -> Forecasts even higher prices -> Votes BUY (1).")
    b.add_p(
        "Result: The trading swarm forms a unanimous 3/3 BUY consensus at the literal exact peak of the manipulation. "
        "Seconds later, the dump crashes the asset, causing catastrophic portfolio drawdown."
    )

    b.add_h2("1.3 The TradeSwarm Breakthrough: Two-Track Architecture with Independent Veto")
    b.add_p(
        "TradeSwarm resolves this vulnerability through a fundamental structural shift: We do not allow the sentiment detector "
        "to participate in the democratic vote. Instead, the architecture is bifurcated into two independent parallel tracks:"
    )

    b.add_code_block(
"""[Market OHLCV Data] ---------> [Market Agent] --------\\
[Telegram Message Text] -----> [Sentiment Agent] -------> [Decision Agent] ---> [Risk Agent] ---> Proposed Trade
[Sequential Price Windows] --> [Prediction Agent] -----/  (Majority Vote)       (Vol/DD Gate)     (BUY/HOLD/SELL)
                                                                                                        |
                                TRACK 2: INDEPENDENT VETO AUTHORITY                                     |
[Normal Baseline Stats] -----\\                                                                         v
(μ_sent, σ_sent, μ_vol, σ_vol) -> [SentimentAnomalyAgent] ---> If Anomaly Index >= θ*:       [THE OVERRIDE LOOP]
[Incoming Telegram Feed] ----/    Composite Z-Score Index         FORCE_REVERSE               Replaces proposed
                                                                  (Overrides Trading Swarm)   BUY with exit!"""
    )

    b.add_p(
        "Track 1 (The Trading Consensus Swarm): Market, Sentiment, and Prediction agents deliberate using standard technical "
        "and momentum principles. Their consensus is aggregated by the Decision Agent and checked by the Risk Agent to produce a Proposed Trade."
    )
    b.add_p(
        "Track 2 (Independent Veto Authority): The SentimentAnomalyAgent operates outside the profit-seeking objective. "
        "It is calibrated strictly on non-manipulative, calm market windows (D_normal). It measures incoming message flow and polarity "
        "against calm baselines using standardized Z-scores. If the composite Anomaly Index breaches threshold θ* (1.25σ), it triggers "
        "a FORCE_REVERSE executive override that supersedes the trading swarm completely."
    )

    # TABLE: PERFORMANCE PROOF
    b.add_h2("1.4 Empirical Verification: Ground Truth & Benchmark Results")
    b.add_p(
        "TradeSwarm was evaluated across a strictly chronological held-out test split (23 real pump-and-dump manipulation episodes "
        "and 22 calm control windows from Binance SYM/BTC pairs):"
    )

    headers = ["Evaluation Metric", "Singh (2026) Baseline", "MSPM Risk Gating (Veto Off)", "TradeSwarm (Veto On)", "Empirical Breakthrough"]
    rows = [
        ["Manipulation Survival Rate", "0.0% (0 / 23 saved)", "0.0% (0 / 23 saved)", "100.0% (23 / 23 saved)", "+100.0% Full Protection"],
        ["Drawdown During Pumps Only", "40.72% loss per pump", "23.73% loss per pump", "0.00% capital lost", "Zero manipulation loss"],
        ["Maximum Portfolio Drawdown", "57.83%", "33.75%", "26.20%", "31.63% absolute risk cut"],
        ["False-Positive Veto Rate", "0.0%", "0.0%", "9.09% (2 / 22 calm)", "Controlled (<10% false alarm)"],
        ["Test Suite Regression Status", "N/A", "N/A", "34 / 34 Tests Passed", "100% verified in 38s"],
    ]
    col_widths = [2600, 2100, 2100, 2100, 2300]
    b.add_table(headers, rows, col_widths)

    b.add_page_break()

    # SECTION 2: DETAILED AGENT-BY-AGENT EXPLANATION
    b.add_h1("2. Elaborate Breakdown of Every Agent")
    b.add_p(
        "Every agent in TradeSwarm inherits from a standardized abstract base class and communicates via strongly typed dataclass signals. "
        "Below is the exact implementation, internal logic, mathematical formulation, and operational role of each component."
    )

    # AGENT 0: BASE AGENT
    b.add_h2("2.1 The Architectural Contract: BaseAgent & AgentSignal")
    b.add_p("File Reference: src/agents/base_agent.py", bold_prefix="Source File:")
    b.add_p(
        "To ensure modularity (inspired by Huang & Tanaka's MSPM design pattern), every agent implements a uniform interface. "
        "Agents accept an environment state dictionary and return an AgentSignal object."
    )
    b.add_code_block(
"""@dataclass
class AgentSignal:
    agent_name: str
    action: Literal['BUY', 'HOLD', 'SELL', 'FORCE_HEDGE', 'FORCE_REVERSE']
    confidence: float   # [0.0, 1.0]
    vote: int           # 1 for BUY, 0 for HOLD/SELL
    raw_value: float    # continuous metric (momentum, polarity, forecast, z-score)
    metadata: Dict[str, Any]

class BaseAgent(ABC):
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> AgentSignal:
        pass"""
    )
    b.add_bullet("Clean Separation of Concerns:", "The orchestrator does not need to know internal agent details; it queries standard signals.")
    b.add_bullet("Metadata Preservation:", "Each agent outputs its internal calculations (e.g., Z-scores, momentum values, message counts) for complete auditability.")

    # AGENT 1: MARKET AGENT
    b.add_h2("2.2 Track 1: MarketAgent (Technical Momentum & Volatility)")
    b.add_p("File Reference: src/agents/market_agent.py", bold_prefix="Source File:")
    b.add_p("Base Paper: Khushvir Singh (2026), Table II Technical Indicators", bold_prefix="Theoretical Foundation:")
    b.add_p(
        "The MarketAgent inspects sequential OHLCV (Open, High, Low, Close, Volume) candlestick data to detect short-term price breakout momentum "
        "while filtering out choppy market noise."
    )
    b.add_p("Mathematical Formulation:", bold_prefix="Formulas:")
    b.add_bullet("1-Period Returns:", "R_t = (Close_t - Close_{t-1}) / Close_{t-1}")
    b.add_bullet("5-Period Momentum:", "M_t^{(5)} = (Close_t - Close_{t-5}) / Close_{t-5}")
    b.add_bullet("10-Period Rolling Volatility:", "σ_t^{(10)} = StdDev(R_{t-9 : t})")
    b.add_bullet("Decision Rule:", "vote = 1 (BUY) if M_t^{(5)} > 0 and σ_t^{(10)} < MedianVol; else vote = 0 (HOLD/SELL)")
    b.add_p(
        "Behavior During Manipulation: When syndicate organizers place large market buy orders to initiate a pump, the price spikes upwards. "
        "The 5-period momentum shoots into positive territory (M_t > 0). Consequently, the MarketAgent issues a BUY signal with high confidence, "
        "unaware that this momentum is artificially manufactured."
    )

    # AGENT 2: SENTIMENT AGENT
    b.add_h2("2.3 Track 1: SentimentAgent (FinBERT & NLP Polarity Scorer)")
    b.add_p("File Reference: src/agents/sentiment_agent.py", bold_prefix="Source File:")
    b.add_p("Base Paper: ProsusAI FinBERT & Ardia & Bluteau (2024) Telegram Corpus", bold_prefix="Theoretical Foundation:")
    b.add_p(
        "The SentimentAgent analyzes social media and Telegram chat streams. It extracts polarity by running text through "
        "either a full FinBERT transformer model (ProsusAI/finbert) or an aligned crypto-lexicon lookup table."
    )
    b.add_bullet("Polarity Formulation:", "Polarity S_t = P(Positive) - P(Negative) ∈ [-1.0, +1.0]")
    b.add_bullet("Lexical Calibration:", "Terms like 'moon', 'rocket', 'target 300%', 'pump' carry weights +0.80 to +0.90; 'dump', 'scam', 'rekt' carry -0.90 to -0.95.")
    b.add_bullet("Decision Rule:", "vote = 1 (BUY) if S_t > 0.05; else vote = 0 (HOLD/SELL)")
    b.add_p(
        "Behavior During Manipulation: Organizers flood channels with hyper-enthusiastic text ('BUY QUICK!', 'WE ARE PUMPING TO THE MOON!'). "
        "FinBERT evaluates this as overwhelming bullish enthusiasm (S_t approaches +0.85). The SentimentAgent enthusiastically votes BUY (1), "
        "falling directly into the syndicate's trap."
    )

    # AGENT 3: PREDICTION AGENT
    b.add_h2("2.4 Track 1: PredictionAgent (2-Layer Stacked PyTorch LSTM)")
    b.add_p("File Reference: src/agents/prediction_agent.py & src/models/lstm_predictor.py", bold_prefix="Source File:")
    b.add_p("Base Paper: Khushvir Singh (2026), Section 3.2 & Table I Architecture", bold_prefix="Theoretical Foundation:")
    b.add_p(
        "The PredictionAgent uses a sequential Deep Learning model to forecast the next price step P̂_{t+1} from a sliding window "
        "of the past 10 closing prices [P_{t-9}, ..., P_t]."
    )
    b.add_bullet("Network Architecture:", "2 stacked LSTM layers, 32 hidden dimensions per layer, Dropout = 0.20, followed by a linear projection layer: ℝ^32 -> ℝ^1.")
    b.add_bullet("Training Protocol:", "Min-max normalization, Adam optimizer (lr = 0.001), Mean Squared Error (MSE) loss, trained over 15 epochs on chronological training splits.")
    b.add_bullet("Decision Rule:", "vote = 1 (BUY) if P̂_{t+1} > P_t; else vote = 0 (HOLD)")
    b.add_p(
        "Behavior During Manipulation: LSTMs extrapolate sequential patterns. When confronted with the sharp upward slope of the pump's initial candle, "
        "the LSTM's hidden state vector projects the trajectory upward, predicting P̂_{t+1} > P_t. Hence, the PredictionAgent votes BUY (1)."
    )

    b.add_page_break()

    # AGENT 4: DECISION AGENT
    b.add_h2("2.5 Track 1: DecisionAgent (Majority Voting & Q-Learning Refinement)")
    b.add_p("File Reference: src/agents/decision_agent.py & src/models/q_learning.py", bold_prefix="Source File:")
    b.add_p("Base Paper: Singh (2026) 4-Agent Majority Voting & Tabular Q-Table", bold_prefix="Theoretical Foundation:")
    b.add_p(
        "The DecisionAgent aggregates the binary votes from the three upstream trading agents: Market (v_m), Sentiment (v_s), and Prediction (v_p)."
    )
    b.add_bullet("Standard Majority Voting Truth Table:", "Sum of Votes = v_m + v_s + v_p ∈ {0, 1, 2, 3}")
    b.add_bullet("Consensus Rules:", "If Sum >= 2: Action = 'BUY' | If Sum == 1: Action = 'HOLD' | If Sum == 0: Action = 'SELL'")
    b.add_bullet("Optional Tabular Q-Learning Refinement:", "Singh's Q-table maps the 3-bit state index (s = 4*v_m + 2*v_s + v_p ∈ [0..7]) to discrete actions using Bellman updates: Q(s, a) <- Q(s, a) + α [R + γ max Q(s', a') - Q(s, a)].")
    b.add_callout(
        "THE DEMOCRATIC TRAP: During a pump, v_m = 1, v_s = 1, v_p = 1. Total votes = 3. "
        "The DecisionAgent issues a 100% BUY recommendation. Every conventional voting system buys the top!",
        "THE ARCHITECTURAL VULNERABILITY"
    )

    # AGENT 5: RISK AGENT
    b.add_h2("2.6 Track 1: RiskAgent (Dynamic Volatility Gating & Drawdown Stop-Loss)")
    b.add_p("File Reference: src/agents/risk_agent.py", bold_prefix="Source File:")
    b.add_p("Base Paper: Huang & Tanaka (2021) MSPM Risk Gating (Eq. 8)", bold_prefix="Theoretical Foundation:")
    b.add_p(
        "The RiskAgent acts as an operational risk controller. It evaluates the proposed trade from the DecisionAgent against portfolio constraints:"
    )
    b.add_bullet("Volatility Gating:", "If trailing 15-period return volatility σ_t exceeds max_volatility (0.05), aggressive BUY actions are gated to HOLD.")
    b.add_bullet("Drawdown Stop-Loss:", "If portfolio trailing drawdown DD_t breaches max_drawdown_limit (0.25), all actions are forced to SELL to protect capital.")
    b.add_bullet("Dynamic Position Sizing:", "Outputs capital deployment multiplier: Size = max(0.2, 1.0 - AggregateRisk).")
    b.add_p(
        "Limitation in Pumps: While the RiskAgent cuts overall market volatility, it is reactive. By the time return volatility spikes, "
        "the pump has already occurred, and the bot cannot distinguish between a legitimate high-volume breakout and a fraudulent pump."
    )

    # AGENT 6: SENTIMENT ANOMALY AGENT
    b.add_h2("2.7 Track 2 (⭐ The Core Novelty): SentimentAnomalyAgent (The Independent Veto Authority)")
    b.add_p("File Reference: src/agents/sentiment_anomaly_agent.py", bold_prefix="Source File:")
    b.add_p("Patent / Novel Formulation: Independent Statistical Anomaly Override Loop", bold_prefix="Theoretical Foundation:")
    b.add_p(
        "The SentimentAnomalyAgent is the defining scientific breakthrough of TradeSwarm. It differs from every other agent in three ways:"
    )
    b.add_bullet("1. No Trading Incentives:", "It does not share a reward function with profit-seeking agents. It never learns to 'chase' quick gains.")
    b.add_bullet("2. Calibrated Strictly on Normal Windows (D_normal):", "During training, it calculates baseline mean and standard deviation of sentiment polarity and message frequency exclusively on calm periods: (μ_sent, σ_sent, μ_vol, σ_vol).")
    b.add_bullet("3. Standardized One-Sided Anomaly Index:", "Measures how many standard deviations current Telegram activity deviates from normal equilibrium:")
    
    b.add_code_block(
"""Formula for Composite Anomaly Index A_t:

Z_sent = (S_t - μ_sent) / max(σ_sent, ε)
Z_vol  = (V_t - μ_vol)  / max(σ_vol, ε)

Z_sent^+ = max(0.0, Z_sent)   # Only positive hype surges count
Z_vol^+  = max(0.0, Z_vol)    # Only sudden message spikes count

Anomaly Index A_t = w_sent * Z_sent^+ + w_vol * Z_vol^+
(Weights: w_sent = 0.60, w_vol = 0.40)"""
    )

    b.add_bullet("Zero-Leakage Optimal Threshold Tuning:", "The critical threshold θ* was tuned strictly on the held-out validation dataset (val.parquet) to maximize F1-score across 45 events. The optimal threshold found was θ* = 1.25.")
    b.add_bullet("Executive Override Policy:", "If A_t >= θ* (1.25σ): Issue FORCE_REVERSE with is_veto = True. Otherwise: Emit passive HOLD (allow Track 1 consensus to pass through).")

    b.add_callout(
        "When the syndicate broadcasts hype, message volume jumps from 4 messages/min to 85 messages/min (Z_vol > 20σ), "
        "and polarity jumps to +0.85 (Z_sent > 4σ). The Anomaly Index spikes to 12.8σ (far above 1.25σ). "
        "The Veto Agent instantly overrides the 3/3 BUY consensus, forcing an emergency exit or capital freeze before the dump hits!",
        "THE VETO INTERCEPTION IN ACTION"
    )

    # AGENT 7: ORCHESTRATOR
    b.add_h2("2.8 Central Engine: TradeSwarmOrchestrator & The Override Loop")
    b.add_p("File Reference: src/orchestrator.py", bold_prefix="Source File:")
    b.add_p(
        "The Orchestrator coordinates both tracks in lockstep at every timestep t:"
    )
    b.add_bullet("Step 1 (Parallel Deliberation):", "Broadcasts market state to MarketAgent, SentimentAgent, and PredictionAgent.")
    b.add_bullet("Step 2 (Consensus Aggregation):", "DecisionAgent tallies votes -> Proposed Action.")
    b.add_bullet("Step 3 (Risk Filter):", "RiskAgent inspects portfolio drawdown and volatility -> Gated Action.")
    b.add_bullet("Step 4 (Anomaly Evaluation):", "SentimentAnomalyAgent evaluates incoming message Z-scores -> Anomaly Index A_t.")
    b.add_bullet("Step 5 (The Override Branch):", "If VETO_ENABLED and A_t >= θ*: Final Action = FORCE_REVERSE. An OverrideEvent is recorded in the audit trail. Else: Final Action = Gated Action.")

    b.add_page_break()

    # SECTION 3: SUMMARY TABLE OF AGENT SPECIFICATIONS
    b.add_h1("3. Comprehensive Agent Specification Matrix")
    b.add_p("Quick reference mapping inputs, outputs, models, and behavior under manipulation for all agents:")

    agent_headers = ["Agent Name", "Track", "Input Data", "Internal Model", "Normal Action", "During Manipulation"]
    agent_rows = [
        ["MarketAgent", "Track 1", "OHLCV Candlesticks", "5-period Momentum, 10-period Volatility", "BUY on calm breakout", "Tricked: Votes BUY (1)"],
        ["SentimentAgent", "Track 1", "Telegram Message Text", "FinBERT Transformer / Crypto Lexicon", "BUY on genuine positive news", "Tricked: Votes BUY (1)"],
        ["PredictionAgent", "Track 1", "Sequential Close Prices", "2-Layer PyTorch Stacked LSTM", "BUY if P̂_{t+1} > P_t", "Tricked: Votes BUY (1)"],
        ["DecisionAgent", "Track 1", "3 Binary Signals", "Majority Rule (≥2 votes) + Q-Table", "Executes swarm consensus", "Tricked: Issues 3/3 BUY"],
        ["RiskAgent", "Track 1", "Proposed Trade + Portfolio", "MSPM Volatility & Drawdown Gating", "Scales position sizing", "Reactive: Cannot prevent pump entry"],
        ["SentimentAnomalyAgent", "Track 2", "Telegram Frequency & Polarity", "Statistical Z-Score Baseline Engine", "Passive HOLD (Pass through)", "HERO: Issues FORCE_REVERSE (Veto)"],
        ["TradeSwarmOrchestrator", "Engine", "All Agent Signals", "Two-Track Override Loop Pipeline", "Passes through proposed trades", "Executes Veto; Saves Portfolio"],
    ]
    col_w_spec = [1900, 1100, 2000, 2400, 2000, 2200]
    b.add_table(agent_headers, agent_rows, col_w_spec)

    b.add_h1("4. Key Takeaways for Academic Viva & Examiners")
    b.add_bullet("Core Question:", "'Why not just train the sentiment agent better?' -> Because organizers intentionally generate textbook bullish text. No sentiment classifier can distinguish genuine excitement from manufactured hype based on words alone. The only tell is statistical departure from calm equilibrium.")
    b.add_bullet("Core Contribution:", "TradeSwarm is the first multi-agent cryptocurrency trading architecture that elevates sentiment anomaly detection from a passive voting feature to an independent executive veto authority.")
    b.add_bullet("Zero Lookahead Leakage:", "Strict chronological train/val/test splits (2017 to 2019). The veto agent was calibrated only on train normal windows, threshold θ* was tuned only on validation, and evaluated on unseen future test pumps.")

    # Write files
    b.build_docx(OUTPUT_FILE_1)
    b.build_docx(OUTPUT_FILE_2)
    print(f"Successfully generated Word files at:\n - {OUTPUT_FILE_1}\n - {OUTPUT_FILE_2}")


if __name__ == "__main__":
    generate_full_guide()

