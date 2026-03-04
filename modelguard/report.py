"""
ModelGuard Report Generator
Unified report that combines all analysis modules.
Supports text summary and HTML export.
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from modelguard.weight_drift import WeightDriftReport
from modelguard.prediction_shift import PredictionShiftReport
from modelguard.activation_drift import ActivationDriftReport
from modelguard.feature_drift import FeatureDriftReport
from modelguard.utils import classify_drift


@dataclass
class DiffReport:
    """
    Unified diff report combining all analysis modules.

    This is the main object returned by compare_models().
    """
    weight_report: Optional[WeightDriftReport] = None
    prediction_report: Optional[PredictionShiftReport] = None
    activation_report: Optional[ActivationDriftReport] = None
    feature_report: Optional[FeatureDriftReport] = None
    model_a_name: str = "Model A"
    model_b_name: str = "Model B"

    def summary(self) -> str:
        """Print complete summary of all analyses."""
        lines = []

        lines.append("\n" + "🔷" * 35)
        lines.append("  🛡️  MODELGUARD — COMPLETE DIFF REPORT")
        lines.append("🔷" * 35)

        lines.append(f"\n  Comparing: {self.model_a_name} vs {self.model_b_name}")

        # --- Overall Stability Score ---
        overall = self._compute_overall_score()
        lines.append(f"\n  ╔══════════════════════════════════════╗")
        lines.append(f"  ║  OVERALL STABILITY SCORE: {overall['score']:.4f}      ║")
        lines.append(f"  ║  STATUS: {overall['level']:<28} ║")
        lines.append(f"  ╚══════════════════════════════════════╝")

        # --- Component Scores ---
        lines.append(f"\n  Component Breakdown:")
        lines.append(f"  {'─' * 50}")

        if self.weight_report:
            score = self.weight_report.overall_drift_score
            level = self.weight_report.overall_drift_level
            bar = self._make_bar(score)
            lines.append(f"  Weight Drift:     {bar} {score:.4f} {level}")

        if self.prediction_report:
            score = self.prediction_report.prediction_drift_score
            level = self.prediction_report.prediction_drift_level
            bar = self._make_bar(score)
            lines.append(f"  Prediction Shift: {bar} {score:.4f} {level}")

        if self.activation_report:
            score = self.activation_report.overall_activation_drift
            level = self.activation_report.overall_drift_level
            bar = self._make_bar(score)
            lines.append(f"  Activation Drift: {bar} {score:.4f} {level}")

        if self.feature_report:
            score = self.feature_report.overall_feature_drift
            level = self.feature_report.overall_drift_level
            bar = self._make_bar(score)
            lines.append(f"  Feature Drift:    {bar} {score:.4f} {level}")

        lines.append(f"  {'─' * 50}")

        # --- Quick Insights ---
        insights = self._generate_insights()
        if insights:
            lines.append(f"\n  💡 Key Insights:")
            for insight in insights:
                lines.append(f"     • {insight}")

        lines.append("\n" + "🔷" * 35 + "\n")

        output = "\n".join(lines)
        print(output)
        return output

    def layer_drift(self) -> str:
        """Show detailed weight drift analysis."""
        if self.weight_report:
            output = self.weight_report.summary()
            print(output)
            return output
        else:
            msg = "⚠️ Weight drift analysis not available."
            print(msg)
            return msg

    def prediction_shift(self) -> str:
        """Show detailed prediction shift analysis."""
        if self.prediction_report:
            output = self.prediction_report.summary()
            print(output)
            return output
        else:
            msg = "⚠️ Prediction shift analysis not available. Provide dataset."
            print(msg)
            return msg

    def activation_drift(self) -> str:
        """Show detailed activation drift analysis."""
        if self.activation_report:
            output = self.activation_report.summary()
            print(output)
            return output
        else:
            msg = "⚠️ Activation drift analysis not available."
            print(msg)
            return msg

    def feature_sensitivity(self) -> str:
        """Show detailed feature drift analysis."""
        if self.feature_report:
            output = self.feature_report.summary()
            print(output)
            return output
        else:
            msg = "⚠️ Feature drift analysis not available."
            print(msg)
            return msg

    def blame(self) -> str:
        """Show blame analysis — which layer caused the most change."""
        if self.activation_report:
            output = self.activation_report.get_blame()
            print(output)
            return output
        else:
            msg = "⚠️ Blame analysis not available. Provide dataset."
            print(msg)
            return msg

    def fingerprint(self) -> Dict[str, float]:
        """
        Generate Model Stability Fingerprint (MSF).
        A single numerical fingerprint capturing model stability.
        """
        msf = {}

        if self.weight_report:
            msf["weight_drift"] = self.weight_report.overall_drift_score

        if self.prediction_report:
            msf["prediction_drift"] = self.prediction_report.prediction_drift_score
            msf["disagreement_rate"] = self.prediction_report.disagreement_rate
            msf["confidence_change"] = abs(self.prediction_report.confidence_change)

        if self.activation_report:
            msf["activation_drift"] = self.activation_report.overall_activation_drift

        if self.feature_report:
            msf["feature_drift"] = self.feature_report.overall_feature_drift
            msf["feature_concentration_change"] = abs(
                self.feature_report.concentration_change
            )

        # Overall MSF score
        scores = list(msf.values())
        msf["overall_stability"] = 1.0 - (sum(scores) / len(scores)) if scores else 1.0

        print("\n  🔑 Model Stability Fingerprint (MSF):")
        print("  " + "─" * 45)
        for key, value in msf.items():
            bar = self._make_bar(value if key != "overall_stability" else 1.0 - value)
            print(f"  {key:<30} {bar} {value:.4f}")
        print("  " + "─" * 45)

        return msf

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire report to a dictionary."""
        result = {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "overall": self._compute_overall_score(),
        }

        if self.weight_report:
            result["weight_drift"] = {
                "overall_score": self.weight_report.overall_drift_score,
                "level": self.weight_report.overall_drift_level,
                "architecture_match": self.weight_report.architecture_match,
                "layers": [
                    {
                        "name": lr.layer_name,
                        "l2_diff": lr.l2_diff,
                        "cosine_sim": lr.cosine_sim,
                        "drift_score": lr.drift_score,
                    }
                    for lr in self.weight_report.layer_results
                ],
            }

        if self.prediction_report:
            result["prediction_shift"] = {
                "drift_score": self.prediction_report.prediction_drift_score,
                "level": self.prediction_report.prediction_drift_level,
                "disagreement_rate": self.prediction_report.disagreement_rate,
                "avg_confidence_a": self.prediction_report.avg_confidence_model_a,
                "avg_confidence_b": self.prediction_report.avg_confidence_model_b,
                "avg_prob_shift": self.prediction_report.avg_probability_shift,
            }

        if self.activation_report:
            result["activation_drift"] = {
                "overall_score": self.activation_report.overall_activation_drift,
                "level": self.activation_report.overall_drift_level,
                "layers": [
                    {
                        "name": lr.layer_name,
                        "cosine_sim": lr.mean_cosine_similarity,
                        "drift_score": lr.activation_drift_score,
                    }
                    for lr in self.activation_report.layer_results
                ],
            }

        if self.feature_report:
            result["feature_drift"] = {
                "overall_score": self.feature_report.overall_feature_drift,
                "level": self.feature_report.overall_drift_level,
                "features": [
                    {
                        "name": fr.feature_name,
                        "importance_a": fr.importance_model_a,
                        "importance_b": fr.importance_model_b,
                        "change": fr.importance_change,
                    }
                    for fr in self.feature_report.feature_results
                ],
            }

        return result

    def export(self, filepath: str):
        """
        Export report to file.

        Supported formats:
        - .html — Beautiful HTML report
        - .json — Machine-readable JSON
        - .txt — Plain text summary
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".json":
            self._export_json(filepath)
        elif ext == ".html":
            self._export_html(filepath)
        elif ext == ".txt":
            self._export_txt(filepath)
        else:
            raise ValueError(
                f"Unsupported format '{ext}'. Use .html, .json, or .txt"
            )

        print(f"📄 Report exported to: {filepath}")

    def _export_json(self, filepath: str):
        """Export as JSON."""
        data = self.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _export_txt(self, filepath: str):
        """Export as plain text."""
        lines = []
        lines.append(self.summary())

        if self.weight_report:
            lines.append(self.weight_report.summary())
        if self.prediction_report:
            lines.append(self.prediction_report.summary())
        if self.activation_report:
            lines.append(self.activation_report.summary())
        if self.feature_report:
            lines.append(self.feature_report.summary())

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _export_html(self, filepath: str):
        """Export as HTML report."""
        overall = self._compute_overall_score()
        insights = self._generate_insights()

        html = self._build_html(overall, insights)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

    def _build_html(self, overall: Dict, insights: list) -> str:
        """Build complete HTML report."""

        # Determine status color
        score = overall["score"]
        if score < 0.1:
            color = "#2ecc71"
            status_text = "STABLE"
        elif score < 0.3:
            color = "#f39c12"
            status_text = "LOW DRIFT"
        elif score < 0.6:
            color = "#e67e22"
            status_text = "MODERATE DRIFT"
        elif score < 0.8:
            color = "#e74c3c"
            status_text = "HIGH DRIFT"
        else:
            color = "#c0392b"
            status_text = "CRITICAL DRIFT"

        # Build weight drift rows
        weight_rows = ""
        if self.weight_report:
            for lr in self.weight_report.layer_results:
                row_color = self._get_row_color(lr.drift_score)
                weight_rows += f"""
                <tr style="background-color: {row_color}">
                    <td>{lr.layer_name}</td>
                    <td>{lr.param_type}</td>
                    <td>{lr.l2_diff:.4f}</td>
                    <td>{lr.cosine_sim:.4f}</td>
                    <td>{lr.drift_score:.4f}</td>
                    <td>{lr.drift_level}</td>
                </tr>"""

        # Build prediction shift section
        pred_section = ""
        if self.prediction_report:
            pr = self.prediction_report
            class_rows = ""
            for cs in pr.class_shifts:
                class_rows += f"""
                <tr>
                    <td>{cs.class_name}</td>
                    <td>{cs.avg_prob_model_a:.4f}</td>
                    <td>{cs.avg_prob_model_b:.4f}</td>
                    <td>{cs.prob_change:+.4f}</td>
                    <td>{cs.count_model_a}</td>
                    <td>{cs.count_model_b}</td>
                </tr>"""

            pred_section = f"""
            <div class="section">
                <h2>🔮 Prediction Shift Analysis</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{pr.disagreement_rate:.2%}</div>
                        <div class="metric-label">Disagreement Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{pr.avg_confidence_model_a:.4f}</div>
                        <div class="metric-label">Confidence (Model A)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{pr.avg_confidence_model_b:.4f}</div>
                        <div class="metric-label">Confidence (Model B)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{pr.avg_probability_shift:.4f}</div>
                        <div class="metric-label">Avg Probability Shift</div>
                    </div>
                </div>
                <table>
                    <tr>
                        <th>Class</th><th>Prob A</th><th>Prob B</th>
                        <th>Change</th><th>Count A</th><th>Count B</th>
                    </tr>
                    {class_rows}
                </table>
            </div>"""

        # Build activation drift section
        act_section = ""
        if self.activation_report:
            ar = self.activation_report
            act_rows = ""
            for lr in ar.layer_results:
                row_color = self._get_row_color(lr.activation_drift_score)
                act_rows += f"""
                <tr style="background-color: {row_color}">
                    <td>{lr.layer_name}</td>
                    <td>{lr.layer_type}</td>
                    <td>{lr.mean_cosine_similarity:.4f}</td>
                    <td>{lr.mean_l2_distance:.4f}</td>
                    <td>{lr.activation_drift_score:.4f}</td>
                    <td>{lr.drift_level}</td>
                </tr>"""

            blame_text = ar.get_blame().replace("\n", "<br>")

            act_section = f"""
            <div class="section">
                <h2>🧠 Activation Drift Analysis</h2>
                <table>
                    <tr>
                        <th>Layer</th><th>Type</th><th>Cosine Sim</th>
                        <th>L2 Distance</th><th>Drift Score</th><th>Status</th>
                    </tr>
                    {act_rows}
                </table>
                <div class="blame-box">
                    <h3>🔍 Blame Analysis</h3>
                    <pre>{blame_text}</pre>
                </div>
            </div>"""

        # Build feature drift section
        feat_section = ""
        if self.feature_report:
            fr = self.feature_report
            feat_rows = ""
            sorted_features = sorted(
                fr.feature_results,
                key=lambda x: abs(x.importance_change),
                reverse=True,
            )
            for f in sorted_features:
                if f.importance_change > 0.01:
                    arrow = "📈"
                elif f.importance_change < -0.01:
                    arrow = "📉"
                else:
                    arrow = "➡️"

                feat_rows += f"""
                <tr>
                    <td>{f.feature_name}</td>
                    <td>{f.importance_model_a:.4f}</td>
                    <td>{f.importance_model_b:.4f}</td>
                    <td>{f.importance_change:+.4f} {arrow}</td>
                    <td>{f.rank_model_a} → {f.rank_model_b}</td>
                </tr>"""

            feat_section = f"""
            <div class="section">
                <h2>🎯 Feature Influence Drift</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{fr.concentration_model_a:.4f}</div>
                        <div class="metric-label">Concentration (Model A)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{fr.concentration_model_b:.4f}</div>
                        <div class="metric-label">Concentration (Model B)</div>
                    </div>
                </div>
                <table>
                    <tr>
                        <th>Feature</th><th>Importance A</th><th>Importance B</th>
                        <th>Change</th><th>Rank Change</th>
                    </tr>
                    {feat_rows}
                </table>
            </div>"""

        # Build insights
        insights_html = ""
        for insight in insights:
            insights_html += f"<li>{insight}</li>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelGuard Diff Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1923;
            color: #e0e0e0;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            color: #ffffff;
        }}
        h2 {{
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #4fc3f7;
            border-bottom: 2px solid #4fc3f7;
            padding-bottom: 5px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #1a2a3a, #0f1923);
            border-radius: 15px;
            border: 1px solid #2a3a4a;
        }}
        .overall-score {{
            display: inline-block;
            padding: 20px 40px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.8em;
            font-weight: bold;
            background: {color}22;
            border: 2px solid {color};
            color: {color};
        }}
        .section {{
            background: #1a2a3a;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #2a3a4a;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: #0f1923;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 1px solid #2a3a4a;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #4fc3f7;
        }}
        .metric-label {{
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th {{
            background: #0f1923;
            color: #4fc3f7;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #2a3a4a;
        }}
        tr:hover {{ background: #253545 !important; }}
        .blame-box {{
            margin-top: 20px;
            padding: 15px;
            background: #0f1923;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
        }}
        .blame-box pre {{
            color: #e0e0e0;
            font-size: 0.9em;
            white-space: pre-wrap;
        }}
        .insights {{
            background: #1a2a3a;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #2a3a4a;
        }}
        .insights li {{
            margin: 8px 0;
            padding-left: 10px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ ModelGuard Diff Report</h1>
            <p>{self.model_a_name} vs {self.model_b_name}</p>
            <div class="overall-score">
                Stability Score: {overall['score']:.4f} — {status_text}
            </div>
        </div>

        <div class="section">
            <h2>📊 Weight Drift Analysis</h2>
            <table>
                <tr>
                    <th>Layer</th><th>Type</th><th>L2 Diff</th>
                    <th>Cosine Sim</th><th>Drift Score</th><th>Status</th>
                </tr>
                {weight_rows}
            </table>
        </div>

        {pred_section}
        {act_section}
        {feat_section}

        <div class="insights">
            <h2>💡 Key Insights</h2>
            <ul>{insights_html}</ul>
        </div>

        <div class="footer">
            Generated by ModelGuard v0.1.0 | github.com/yourname/modelguard
        </div>
    </div>
</body>
</html>"""
        return html

    def _get_row_color(self, score: float) -> str:
        """Get background color based on drift score."""
        if score < 0.1:
            return "transparent"
        elif score < 0.3:
            return "#f39c1215"
        elif score < 0.6:
            return "#e67e2220"
        elif score < 0.8:
            return "#e74c3c20"
        else:
            return "#c0392b30"

    def _compute_overall_score(self) -> Dict[str, Any]:
        """Compute overall stability score."""
        scores = []
        weights = []

        if self.weight_report:
            scores.append(self.weight_report.overall_drift_score)
            weights.append(0.25)

        if self.prediction_report:
            scores.append(self.prediction_report.prediction_drift_score)
            weights.append(0.35)

        if self.activation_report:
            scores.append(self.activation_report.overall_activation_drift)
            weights.append(0.25)

        if self.feature_report:
            scores.append(self.feature_report.overall_feature_drift)
            weights.append(0.15)

        if scores:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            overall = sum(s * w for s, w in zip(scores, normalized_weights))
        else:
            overall = 0.0

        return {
            "score": overall,
            "level": classify_drift(overall),
        }

    def _make_bar(self, score: float) -> str:
        """Create a visual bar for a score."""
        filled = int(score * 10)
        empty = 10 - filled
        return "█" * filled + "░" * empty

    def _generate_insights(self) -> list:
        """Generate human-readable insights from all analyses."""
        insights = []

        # Weight insights
        if self.weight_report:
            wr = self.weight_report
            if not wr.architecture_match:
                insights.append(
                    "⚠️ Model architectures don't match — "
                    "some layers couldn't be compared."
                )
            if wr.most_drifted_layers:
                top = wr.most_drifted_layers[0]
                insights.append(
                    f"Highest weight drift in layer '{top}'."
                )

        # Prediction insights
        if self.prediction_report:
            pr = self.prediction_report
            if pr.disagreement_rate > 0.1:
                insights.append(
                    f"🔴 {pr.disagreement_rate:.1%} of predictions changed — "
                    f"significant behavioral shift."
                )
            elif pr.disagreement_rate > 0.01:
                insights.append(
                    f"⚠️ {pr.disagreement_rate:.1%} of predictions changed."
                )
            else:
                insights.append(
                    "✅ Predictions are consistent between models."
                )

            if abs(pr.confidence_change) > 0.05:
                direction = "increased" if pr.confidence_change > 0 else "decreased"
                insights.append(
                    f"Model confidence {direction} by "
                    f"{abs(pr.confidence_change):.4f} on average."
                )

            # Class shift insights
            for cs in pr.class_shifts:
                if abs(cs.prob_change) > 0.05:
                    direction = "gained" if cs.prob_change > 0 else "lost"
                    insights.append(
                        f"Class '{cs.class_name}' {direction} "
                        f"{abs(cs.prob_change):.4f} average probability."
                    )

        # Activation insights
        if self.activation_report:
            ar = self.activation_report
            if ar.most_drifted_layers:
                top = ar.most_drifted_layers[0]
                insights.append(
                    f"Internal representations changed most at layer '{top}'."
                )

            # Dead neuron changes
            for lr in ar.layer_results:
                if lr.dead_neuron_change > 2:
                    insights.append(
                        f"⚠️ Layer '{lr.layer_name}': "
                        f"{lr.dead_neuron_change} neurons became inactive."
                    )

        # Feature insights
        if self.feature_report:
            fr = self.feature_report
            if fr.top_gained:
                insights.append(
                    f"Model now relies MORE on: {', '.join(fr.top_gained[:3])}."
                )
            if fr.top_lost:
                insights.append(
                    f"Model now relies LESS on: {', '.join(fr.top_lost[:3])}."
                )
            if abs(fr.concentration_change) > 0.05:
                direction = "more concentrated" if fr.concentration_change > 0 else "more distributed"
                insights.append(
                    f"Feature importance became {direction}."
                )

        return insights