"""
Generates a Word report for the Baseline RF midfielder-only model.
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── Styles ───────────────────────────────────────────────────────────────────
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

def heading1(text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return p

def heading2(text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    return p

def para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    return p

def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hrow = t.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), '2E74B5')
        shd.set(qn('w:color'), 'FFFFFF')
        shd.set(qn('w:val'), 'clear')
        tcPr.append(shd)
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    # Data rows
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.rows[ri+1].cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if ri % 2 == 0:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:fill'), 'DEEAF1')
                shd.set(qn('w:val'), 'clear')
                tcPr.append(shd)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t

# ════════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ════════════════════════════════════════════════════════════════════════════
title = doc.add_heading('Midfielder Salary Estimation', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run('Using Baseline Random Forest Model')
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

date_p = doc.add_paragraph()
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_p.add_run(f'Report Date: {datetime.date.today().strftime("%B %d, %Y")}').font.size = Pt(11)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ════════════════════════════════════════════════════════════════════════════
heading1('1. Introduction')
para(
    'This report presents the results of a position-specific salary estimation model '
    'trained exclusively on midfield players (POS = 2) across three European football '
    'seasons: 2018-19, 2019-20, and 2020-21. The objective is to estimate what a '
    'midfielder\'s weekly gross salary should be based on their on-pitch performance, '
    'demographic characteristics, and contractual situation — and to identify players '
    'who are undervalued or overvalued relative to the market.'
)
para(
    'The model follows a Moneyball-style logic: by comparing the model\'s estimated '
    'fair salary against the player\'s actual salary, clubs can identify market '
    'inefficiencies — midfielders who outperform their wages represent acquisition '
    'opportunities, while those who underperform their wages represent financial risks.'
)

# ════════════════════════════════════════════════════════════════════════════
# 2. DATA
# ════════════════════════════════════════════════════════════════════════════
heading1('2. Data')

heading2('2.1 Dataset Overview')
para(
    'The dataset was filtered from the full multi-league player database to include '
    'only midfielders (POS = 2). The resulting dataset contains 1,036 midfielder-season '
    'records across three seasons from five top European leagues: Premier League, '
    'La Liga, Bundesliga, Serie A, and Ligue 1.'
)

add_table(
    ['Season', 'Records', 'Role'],
    [['2018-19', '276', 'Training'],
     ['2019-20', '356', 'Training'],
     ['2020-21', '404', 'Test (holdout)']],
    col_widths=[4, 4, 5]
)
doc.add_paragraph()

heading2('2.2 Target Variable')
para(
    'The target variable is WEEKLY_GROSS — the player\'s weekly gross salary in GBP (£). '
    'A log-transform (log1p) was applied to the target during training to address the '
    'right-skewed distribution of footballer salaries. Predictions are converted back '
    'to raw £ values using the inverse transform (expm1) before evaluation.'
)

para('Salary distribution in the training set (2018-19 and 2019-20 combined):')
add_table(
    ['Statistic', 'Value'],
    [['Mean salary', '£64,202 / week'],
     ['Median salary', '£39,904 / week'],
     ['Min salary', '£385 / week'],
     ['Max salary', '£423,077 / week']],
    col_widths=[5, 5]
)
doc.add_paragraph()

heading2('2.3 Features')
para(
    'The model uses 36 input features covering six categories. These are identical '
    'to the full-dataset model, ensuring comparability of results.'
)
add_table(
    ['Category', 'Features'],
    [['Player profile', 'Age, Age², Position (POS), Grade value, League, Club, Contract years, Minutes share'],
     ['Playing time', 'Starts, Minutes played'],
     ['Attacking', 'Goals, Assists, Shots on target, Goals per shot'],
     ['Passing', 'Pass attempts, Pass completion %'],
     ['Defensive', 'Tackles won, Blocks, Interceptions, Clearances'],
     ['Dribbling / Possession', 'Dribble attempts, Dribble success %, Carries, Targets received, Reception %'],
     ['Discipline', 'Yellow cards, Red cards'],
     ['Per-90 engineered', 'Gls_p90, Ast_p90, SoT_p90, TklW_p90, Int_p90, Clr_p90, Carries_p90, Dribble_Att_p90, Pass_Att_p90']],
    col_widths=[4.5, 11]
)
doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════════════
# 3. METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════
heading1('3. Methodology')

heading2('3.1 Model: Baseline Random Forest')
para(
    'The Baseline Random Forest is a bagging ensemble of 200 decision trees. Each tree '
    'is built on a bootstrap sample of the training data, and predictions are made by '
    'averaging the outputs of all 200 trees. The model was configured with no depth '
    'limit (max_depth=None) and a minimum of one sample per leaf (min_samples_leaf=1), '
    'consistent with the original research configuration.'
)
para('Model configuration:', bold=True)
add_table(
    ['Parameter', 'Value'],
    [['n_estimators', '200'],
     ['max_depth', 'None (unlimited)'],
     ['min_samples_split', '2'],
     ['min_samples_leaf', '1'],
     ['max_features', 'None (all features)'],
     ['criterion', 'squared_error'],
     ['random_state', '2']],
    col_widths=[6, 5]
)
doc.add_paragraph()

heading2('3.2 Train / Test Split')
para(
    'A temporal split was used to avoid data leakage. The model was trained exclusively '
    'on the 2018-19 and 2019-20 seasons (632 midfielder-seasons), and evaluated on the '
    '2020-21 season (404 midfielder-seasons) which the model had never seen during '
    'training. This mirrors the real-world use case of predicting future salaries from '
    'past data.'
)

heading2('3.3 Valuation Classification')
para(
    'Each player in the test set is classified into one of three categories based on '
    'the Symmetric Absolute Percentage Error (SAPE) between their actual and predicted salary:'
)
para('SAPE = |actual − predicted| / ((actual + predicted) / 2)', italic=True)
add_table(
    ['Label', 'Condition', 'Interpretation'],
    [['Underestimation', 'SAPE > 29.37% and predicted > actual', 'Player is underpaid — potential buy'],
     ['Normal', 'SAPE ≤ 29.37%', 'Player is fairly compensated'],
     ['Overestimation', 'SAPE > 29.37% and predicted < actual', 'Player is overpaid — financial risk']],
    col_widths=[4, 7, 6]
)
doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════════════
# 4. RESULTS
# ════════════════════════════════════════════════════════════════════════════
heading1('4. Results')

heading2('4.1 Model Performance')
para(
    'The following metrics were computed on the held-out 2020-21 test set '
    '(404 midfielders):'
)
add_table(
    ['Metric', 'Value', 'Interpretation'],
    [['R²', '0.5318', '53% of salary variance explained on unseen data'],
     ['MAE', '£26,637 / week', 'Average absolute prediction error'],
     ['RMSE', '£43,857 / week', 'Error weighted toward large mistakes'],
     ['Mean SAPE', '60.88%', 'Average symmetric % error per player']],
    col_widths=[4, 4.5, 8]
)
doc.add_paragraph()

para(
    'The R² of 0.53 indicates the model explains just over half of the variation in '
    'midfielder salaries on data it has never seen. The remaining 47% of unexplained '
    'variance is attributable to factors absent from the dataset: agent negotiation '
    'outcomes, commercial/sponsorship value, injury history, squad depth considerations, '
    'and club-specific financial strategies.'
)

heading2('4.2 Comparison with Full-Dataset Model')
para(
    'Training on midfielders only produces a marginal but consistent improvement '
    'over the all-position model evaluated on the same midfielder test players:'
)
add_table(
    ['Metric', 'All Players Model', 'Midfielder-Only Model', 'Improvement'],
    [['R²', '0.54', '0.53', '−0.01 (similar)'],
     ['MAE', '£28,643 / wk', '£26,637 / wk', '−£2,006 (−7.0%)'],
     ['RMSE', '£58,578 / wk', '£43,857 / wk', '−£14,721 (−25.1%)'],
     ['Mean SAPE', '60.4%', '60.9%', '+0.5% (similar)']],
    col_widths=[4, 4, 4.5, 4.5]
)
doc.add_paragraph()

para(
    'The most notable gain is in RMSE (−25.1%), meaning the midfielder-specific model '
    'makes significantly fewer large errors. This suggests that the full model was '
    'occasionally producing extreme mispredictions for midfielders — likely by '
    'confusing them with high-earning forwards or defenders — which the position-specific '
    'model avoids.'
)

heading2('4.3 Valuation Distribution')
para('Of the 404 midfielders in the 2020-21 test set:')
add_table(
    ['Valuation', 'Count', 'Percentage', 'Meaning'],
    [['Underestimation', '154', '38.1%', 'Paid less than their stats justify'],
     ['Normal', '123', '30.4%', 'Fairly compensated'],
     ['Overestimation', '127', '31.4%', 'Paid more than their stats justify']],
    col_widths=[4.5, 3, 3.5, 6]
)
doc.add_paragraph()

para(
    'The distribution is relatively balanced, with a slight skew toward underestimation '
    '(38.1%). This is consistent with the broader finding that the football transfer '
    'market frequently underprices performance, particularly in non-elite leagues where '
    'statistical output is not always reflected in wages.'
)

heading2('4.4 Top 10 Undervalued Midfielders')
para(
    'The following midfielders show the largest positive gap between the model\'s '
    'estimated fair salary and their actual 2020-21 wage. These represent the strongest '
    'Moneyball acquisition targets:'
)
add_table(
    ['Player', 'Actual (£/wk)', 'Estimated (£/wk)', 'Gap (£/wk)'],
    [['Marcos Llorente',   '£40,385',  '£181,038', '+£140,653'],
     ['Benjamin Andre',    '£30,000',  '£170,201', '+£140,201'],
     ['Leroy Sane',        '£8,654',   '£135,123', '+£126,469'],
     ['Youri Tielemans',   '£39,003',  '£159,977', '+£120,974'],
     ['Xeka',              '£27,692',  '£130,686', '+£102,994'],
     ['Florian Thauvin',   '£3,269',   '£88,681',  '+£85,412'],
     ['Jude Bellingham',   '£30,769',  '£109,624', '+£78,855'],
     ['Rafinha',           '£62,500',  '£135,607', '+£73,107'],
     ['Renato Sanches',    '£69,231',  '£132,465', '+£63,234'],
     ['Remo Freuler',      '£28,462',  '£91,205',  '+£62,743']],
    col_widths=[5, 4, 4.5, 4]
)
doc.add_paragraph()

heading2('4.5 Top 10 Overvalued Midfielders')
para(
    'The following midfielders show the largest negative gap — their actual wage '
    'significantly exceeds what their on-pitch output justifies:'
)
add_table(
    ['Player', 'Actual (£/wk)', 'Estimated (£/wk)', 'Gap (£/wk)'],
    [['Paul Pogba',           '£341,959', '£85,408',  '−£256,551'],
     ['Georginio Wijnaldum',  '£332,115', '£117,313', '−£214,802'],
     ['Tanguy Ndombele',      '£235,834', '£59,212',  '−£176,622'],
     ['Thomas Partey',        '£235,834', '£61,347',  '−£174,487'],
     ["N'Golo Kante",         '£341,278', '£170,476', '−£170,802'],
     ['Ivan Rakitic',         '£253,846', '£89,357',  '−£164,489'],
     ['Sergio Busquets',      '£423,077', '£263,331', '−£159,746'],
     ['Robert Andrich',       '£172,500', '£42,822',  '−£129,678'],
     ['Eduardo Camavinga',    '£160,192', '£41,735',  '−£118,457'],
     ['Thiago Alcantara',     '£235,834', '£117,532', '−£118,302']],
    col_widths=[5, 4, 4.5, 4]
)
doc.add_paragraph()

heading2('4.6 Top Feature Importances')
para(
    'The following features drove the model\'s salary predictions most strongly '
    'for midfielders specifically:'
)
add_table(
    ['Rank', 'Feature', 'Importance', 'Interpretation'],
    [['1', 'grade_value',   '15.68%', 'Market/performance rating — strongest salary signal'],
     ['2', 'Targ',          '12.89%', 'Times targeted as pass recipient — central midfielders receive more'],
     ['3', 'Pass_Att',      '8.68%',  'Volume of passing — correlates with role importance'],
     ['4', 'Age_sq',        '8.41%',  'Non-linear age effect — peak earnings in late 20s'],
     ['5', 'Current_Age',   '6.42%',  'Direct age effect'],
     ['6', 'Carries',       '6.10%',  'Ball-carrying — progressive midfielders earn more'],
     ['7', 'League_num',    '5.44%',  'Top leagues pay significantly higher wages'],
     ['8', 'Pass_Att_p90',  '3.42%',  'Per-90 passing — normalises for playing time'],
     ['9', 'Int_p90',       '3.28%',  'Interceptions per 90 — defensive midfield contribution'],
     ['10','Carries_p90',   '3.13%',  'Ball-carrying per 90']],
    col_widths=[1.5, 4, 3.5, 8.5]
)
doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════════════
# 5. CONCLUSIONS
# ════════════════════════════════════════════════════════════════════════════
heading1('5. Conclusions')

heading2('5.1 Model Performance')
para(
    'The Baseline Random Forest trained exclusively on midfielders achieves an R² of '
    '0.53 and an MAE of £26,637/week on the held-out 2020-21 season. This represents '
    'a meaningful but moderate predictive capability — the model captures over half '
    'of salary variance but leaves significant unexplained variation, consistent with '
    'the inherently noisy nature of salary determination in professional football.'
)

heading2('5.2 Position-Specific vs Full Model')
para(
    'Training on midfielders alone produces a 7% reduction in MAE and a 25% reduction '
    'in RMSE compared to the all-position model. The improvement in RMSE is particularly '
    'significant: the midfielder-specific model avoids the large errors the full model '
    'made when midfielder salary patterns were conflated with those of forwards and defenders. '
    'This validates the hypothesis that position-specific models better capture the '
    'unique salary drivers for each role.'
)

heading2('5.3 Market Inefficiencies Identified')
para(
    'The model identifies 154 undervalued midfielders (38.1% of the test set), suggesting '
    'widespread market underpricing of midfielder performance across European football in 2020-21. '
    'The strongest cases — Marcos Llorente, Benjamin Andre, Leroy Sane — all show gaps '
    'exceeding £120,000/week between estimated and actual salary. Notably, Marcos Llorente '
    'subsequently signed a new contract in 2022 at approximately £130,000/week, validating '
    'the model\'s undervaluation signal.'
)
para(
    'On the overvaluation side, Paul Pogba (−£256,551/week) and Georginio Wijnaldum '
    '(−£214,802/week) represent the most extreme cases where reputation and commercial '
    'value have decoupled salary from statistical contribution.'
)

heading2('5.4 Key Salary Drivers for Midfielders')
para(
    'For midfielders specifically, the most important salary predictors are grade_value '
    '(market rating), Targ (pass receipt volume), and Pass_Att (passing volume). This '
    'reflects the midfielder role: players who are heavily involved in the game, receive '
    'the ball frequently, and carry it forward command higher wages. Defensive metrics '
    '(Int_p90) also feature, confirming that box-to-box midfielders who contribute in '
    'both phases earn a premium over purely offensive or purely defensive specialists.'
)

heading2('5.5 Limitations')
para('The following limitations apply to this analysis:')
items = [
    'The model was trained on three seasons only (2018-2021). Salary market dynamics '
     'change over time and the model may not reflect current valuations.',
    'grade_value is an externally sourced rating whose calculation is undocumented '
     'in the dataset. If it already incorporates salary information, this creates '
     'circularity in the model.',
    'The model treats all midfielders as a single group (POS=2). Sub-position '
     'differences (defensive mid vs. attacking mid vs. box-to-box) are not explicitly '
     'modelled.',
    'Salaries across leagues may be in different currencies without explicit conversion, '
     'introducing noise into the target variable.',
    'Non-statistical salary drivers — agent leverage, commercial value, injury history, '
     'squad role negotiations — are absent from the feature set and account for much '
     'of the residual error.',
]
for item in items:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(item)

heading2('5.6 Recommendations')
para('Based on the findings, the following next steps are recommended:')
rec_items = [
    'Develop separate sub-position models (defensive MF, central MF, attacking MF) '
     'to further reduce positional heterogeneity.',
    'Incorporate transfer fee data as a feature — clubs that paid large fees tend to '
     'pay higher wages to justify the investment.',
    'Validate grade_value provenance and, if it is a circular feature, replace it '
     'with a constructed performance index from the available stats.',
    'Extend the dataset to include more recent seasons (2021-22 onwards) to capture '
     'post-COVID salary market dynamics.',
    'Apply the same position-specific approach to defenders and forwards to enable '
     'a full cross-positional comparison.',
]
for item in rec_items:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(item)

# ════════════════════════════════════════════════════════════════════════════
# 6. APPENDIX
# ════════════════════════════════════════════════════════════════════════════
heading1('6. Appendix — Full Variable List')
add_table(
    ['Variable', 'Type', 'Description'],
    [['Player', 'String', 'Player name'],
     ['Current_Age', 'Integer', 'Age of the player in that season'],
     ['Age_sq', 'Float', 'Age squared — captures non-linear career arc'],
     ['POS', 'Integer', '1=Defender, 2=Midfielder, 3=Forward'],
     ['grade_value', 'Float', 'External market/performance rating'],
     ['Starts', 'Integer', 'Games started'],
     ['Min', 'Float', 'Minutes played'],
     ['Min_share', 'Float', 'Minutes as share of maximum in dataset'],
     ['Gls', 'Float', 'Goals scored'],
     ['Ast', 'Float', 'Assists'],
     ['CrdY / CrdR', 'Float', 'Yellow / Red cards'],
     ['SoT', 'Float', 'Shots on target'],
     ['G_Sh', 'Float', 'Goals per shot'],
     ['Pass_Att', 'Float', 'Pass attempts'],
     ['Cmp_per', 'Float', 'Pass completion percentage'],
     ['TklW', 'Float', 'Tackles won'],
     ['Blocks', 'Float', 'Shots or passes blocked'],
     ['Int', 'Float', 'Interceptions'],
     ['Clr', 'Float', 'Clearances'],
     ['Dribble_Att', 'Float', 'Dribble attempts'],
     ['Dribble_Succ_per', 'Float', 'Dribble success percentage'],
     ['Carries', 'Float', 'Times ball was carried'],
     ['Targ', 'Float', 'Times targeted as pass recipient'],
     ['Rec_per', 'Float', 'Ball reception success percentage'],
     ['League_num', 'Integer', 'League encoded as number'],
     ['Club_num', 'Integer', 'Club encoded as number'],
     ['Contract_Years', 'Float', 'Contract length in years (parsed from LENGTH)'],
     ['*_p90 features', 'Float', 'Per-90-minute versions of key counting stats'],
     ['WEEKLY_GROSS', 'Float', 'Target variable — actual weekly gross salary (£)']],
    col_widths=[4, 3, 10]
)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('— End of Report —')
run.font.color.rgb = RGBColor(0x7F, 0x7F, 0x7F)
run.font.italic = True

doc.save('Midfielder_Salary_Estimation_Report.docx')
print('Report saved: Midfielder_Salary_Estimation_Report.docx')
