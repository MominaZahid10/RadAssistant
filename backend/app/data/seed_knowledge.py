"""
RadAssist AI — Curated Radiology Knowledge Base (Seed Data)

SOURCE ATTRIBUTION:
This knowledge base contains factual medical information compiled from
established, peer-reviewed medical literature and universally accepted
radiological principles. The content represents standard-of-care medical
knowledge found in authoritative sources including:

  • StatPearls (NCBI/NLM) — Peer-reviewed, continuously updated medical reference
  • RadioPaedia — Community-reviewed radiology education resource
  • ACR Appropriateness Criteria — Evidence-based imaging guidelines
  • Felson's Principles of Chest Roentgenology (textbook standard)
  • Fundamentals of Diagnostic Radiology (Brant & Helms)

DISCLAIMER:
This is an educational/research tool. All clinical decisions must be made
by qualified radiologists. This system assists — it does NOT replace
clinical judgment.

STRUCTURE:
Each entry is a dict with:
  - title: Article/topic name
  - source_type: Category for filtering
  - content: The actual medical knowledge text
  - source_attribution: Where this knowledge comes from
"""

# ══════════════════════════════════════════════════════════════
# RADIOLOGY KNOWLEDGE — Organized by body system & modality
# ══════════════════════════════════════════════════════════════

SEED_KNOWLEDGE: list[dict] = [

    # ──────────────────────────────────────────────────────────
    # CHEST RADIOGRAPHY — The most common radiological exam
    # ──────────────────────────────────────────────────────────

    {
        "title": "Systematic Approach to Chest X-Ray Interpretation",
        "source_type": "guideline",
        "source_attribution": "Based on Felson's Principles of Chest Roentgenology and ACR Practice Guidelines",
        "content": """
Systematic Approach to Chest X-Ray (CXR) Interpretation

A systematic approach prevents missed findings. The recommended sequence follows the ABCDEFGHI mnemonic:

A — Airway: Assess the trachea for midline position. Tracheal deviation suggests tension pneumothorax (away from affected side), large pleural effusion (away), or lobar collapse (toward affected side). Check for tracheal narrowing or masses. The carina should be at the level of T5-T7.

B — Bones and Soft Tissues: Examine all visible bony structures including ribs, clavicles, scapulae, humeral heads, and visible spine. Look for fractures (rib fractures are commonly missed), lytic or blastic lesions suggesting metastatic disease, and degenerative changes. Assess soft tissues for subcutaneous emphysema, masses, or calcifications. Check for symmetry of soft tissue shadows.

C — Cardiac Silhouette: The cardiothoracic ratio (CTR) should be less than 0.5 on a PA (posteroanterior) film. A CTR greater than 0.5 suggests cardiomegaly. Note that AP (anteroposterior) films magnify the heart and are unreliable for CTR assessment. Evaluate the cardiac borders: the right heart border is formed by the right atrium, and the left heart border by the left ventricle and left atrial appendage. Loss of distinct cardiac borders (silhouette sign) indicates adjacent consolidation.

D — Diaphragm: The right hemidiaphragm is normally 1-2 cm higher than the left due to the liver. Flattened diaphragms suggest hyperinflation (COPD/emphysema). Free air under the diaphragm (pneumoperitoneum) on an erect CXR indicates bowel perforation until proven otherwise — this is a surgical emergency. Elevated hemidiaphragm may indicate phrenic nerve palsy, hepatomegaly, or subpulmonic effusion.

E — Effusion and Extra-pleural Space: Blunting of the costophrenic angles is the earliest sign of pleural effusion on an erect CXR, requiring approximately 200-300 mL of fluid. A lateral decubitus film can detect as little as 50 mL. Large effusions cause a meniscus sign. Loculated effusions do not shift with position. Empyema shows a lenticular (lens-shaped) collection on CT with split pleura sign.

F — Fields (Lung Fields): Compare both lung fields systematically from apex to base. Look for consolidation (air-space opacification with air bronchograms), ground-glass opacity, masses, nodules, and cavitation. Consolidation with air bronchograms is characteristic of pneumonia. Multiple bilateral opacities suggest pulmonary edema, hemorrhage, or multifocal infection. Unilateral white-out differential includes massive effusion, total lung collapse, or pneumonectomy.

G — Great Vessels and Mediastinum: The mediastinum should not exceed 8 cm in width on a PA film. Widened mediastinum raises concern for aortic dissection, lymphadenopathy, or mass. The aortic knuckle should be well-defined. Loss of the aortic knuckle contour may indicate aortic pathology. Hilar structures should be symmetric; unilateral hilar enlargement suggests lymphadenopathy or pulmonary artery dilation.

H — Hila: The left hilum is normally slightly higher than the right. Bilateral hilar lymphadenopathy (BHL) is the hallmark of sarcoidosis but also seen in lymphoma and metastatic disease. Unilateral hilar prominence may represent a central lung mass or pulmonary embolism with pulmonary artery enlargement.

I — Inserted Lines and Devices: Confirm appropriate positioning of all lines and tubes. ETT (endotracheal tube) tip should be 3-5 cm above the carina. Central venous catheter tip should be at the cavoatrial junction. Chest drain should be in the pleural space. Nasogastric tube should follow the esophagus and cross the diaphragm. Swan-Ganz catheter tip should not extend beyond the proximal pulmonary arteries.

Technical Adequacy Assessment: Before interpretation, confirm the radiograph is technically adequate using the RIPE criteria:
- R (Rotation): Medial ends of clavicles should be equidistant from spinous processes
- I (Inspiration): At least 6 anterior ribs or 10 posterior ribs should be visible above the diaphragm
- P (Penetration/Exposure): Vertebral bodies should be barely visible behind the heart
- E (Erect): Confirm patient positioning, as supine films alter fluid distribution and heart size
"""
    },

    {
        "title": "Pneumonia — Radiographic Findings and Patterns",
        "source_type": "textbook",
        "source_attribution": "Based on StatPearls 'Pneumonia, Imaging' and Fundamentals of Diagnostic Radiology",
        "content": """
Pneumonia — Radiographic Diagnosis and Classification

Definition: Pneumonia is an infection of the lung parenchyma causing inflammation and consolidation. Chest X-ray remains the primary imaging modality for diagnosis, with CT reserved for complicated cases.

Patterns of Pneumonia on Imaging:

1. LOBAR PNEUMONIA (Typical/Community-Acquired):
   - Cause: Most commonly Streptococcus pneumoniae (pneumococcus)
   - CXR Findings: Homogeneous, dense consolidation confined to an anatomical lobe with well-defined fissural borders. Air bronchograms are characteristically present (air-filled bronchi visible within opacified lung). Volume is preserved or slightly increased (unlike atelectasis which causes volume loss).
   - Distribution: Unilateral, single lobe most common. Right lower lobe is the most frequently affected.
   - Complications: Parapneumonic effusion (in 40% of cases), empyema, lung abscess, cavitation.

2. BRONCHOPNEUMONIA (Lobular Pneumonia):
   - Cause: Staphylococcus aureus, Haemophilus influenzae, gram-negative organisms
   - CXR Findings: Patchy, multifocal, bilateral opacities, predominantly in the lower lobes. Opacities are heterogeneous and follow a segmental or subsegmental distribution. Air bronchograms are less common than in lobar pneumonia.
   - Distribution: Often bilateral, patchy, lower lobe predominant.
   - Key Feature: "Tree-in-bud" pattern on CT representing centrilobular nodules and branching opacities from infected bronchioles.

3. INTERSTITIAL PNEUMONIA (Atypical):
   - Cause: Mycoplasma pneumoniae, viruses (influenza, COVID-19, RSV), Chlamydia, Legionella
   - CXR Findings: Diffuse, bilateral reticular or reticulonodular opacities. Ground-glass opacities on CT. Often described as "worse than expected clinically" (clinical-radiological dissociation). Consolidation is less prominent than in typical pneumonia.
   - Distribution: Bilateral, diffuse, often perihilar.
   - COVID-19 Specific: Bilateral, peripheral, posterior ground-glass opacities predominantly in lower lobes. May progress to consolidation. "Crazy paving" pattern (GGO with interlobular septal thickening).

4. ROUND PNEUMONIA:
   - Predominantly seen in children under 8 years old (due to immature collateral ventilation pathways — pores of Kohn and canals of Lambert).
   - CXR Findings: Well-defined round opacity that may mimic a lung mass. Most commonly in posterior segments of lower lobes.
   - Key Point: In a febrile child with a round opacity, treat as pneumonia first and follow up with imaging. Do not assume it is a mass.

5. ASPIRATION PNEUMONIA:
   - CXR Findings: Consolidation in dependent lung segments. In the supine patient: posterior segments of upper lobes and superior segments of lower lobes. In the upright patient: basal segments of lower lobes (right more than left due to the more vertical right main bronchus).

Differential Diagnosis of Consolidation:
- Pneumonia (most common)
- Pulmonary hemorrhage
- Pulmonary edema
- Cryptogenic organizing pneumonia (COP)
- Bronchoalveolar carcinoma / lepidic adenocarcinoma
- Pulmonary infarction

Complications to Evaluate:
- Pleural effusion (parapneumonic vs. empyema)
- Lung abscess (thick-walled cavity with air-fluid level)
- Necrotizing pneumonia
- ARDS (bilateral diffuse opacities with acute onset)

Follow-up Recommendations:
- Uncomplicated pneumonia: Follow-up CXR at 6-8 weeks to confirm resolution
- If opacity persists beyond 12 weeks despite treatment, consider alternative diagnosis (malignancy, TB, organizing pneumonia) and recommend CT
"""
    },

    {
        "title": "Pneumothorax — Types, Imaging, and Management",
        "source_type": "textbook",
        "source_attribution": "Based on StatPearls 'Pneumothorax' and British Thoracic Society Guidelines",
        "content": """
Pneumothorax — Radiographic Diagnosis and Classification

Definition: Pneumothorax is the presence of air in the pleural space, causing partial or complete lung collapse. It is a critical diagnosis that must not be missed on chest imaging.

Types of Pneumothorax:

1. SIMPLE (PRIMARY SPONTANEOUS):
   - Occurs without underlying lung disease, typically in tall, thin young males (age 15-35).
   - Risk factors: Smoking, Marfan syndrome, family history.
   - Mechanism: Rupture of apical subpleural blebs.
   - Usually small and may resolve spontaneously.

2. SECONDARY SPONTANEOUS:
   - Occurs in patients with underlying lung disease (COPD, cystic fibrosis, interstitial lung disease, Pneumocystis pneumonia in HIV).
   - More dangerous than primary because patients have less respiratory reserve.

3. TRAUMATIC:
   - Blunt or penetrating chest trauma, iatrogenic (central line insertion, thoracentesis, lung biopsy, mechanical ventilation).
   - Always look for associated hemothorax (hemopneumothorax) and rib fractures.

4. TENSION PNEUMOTHORAX:
   - MEDICAL EMERGENCY — do not wait for imaging if clinically suspected.
   - Mechanism: One-way valve effect causes progressive air accumulation, compressing the mediastinum and impeding venous return.
   - CXR Findings: Large pneumothorax with contralateral mediastinal shift, flattening of the ipsilateral hemidiaphragm, and compression of the contralateral lung. Widened intercostal spaces on the affected side.
   - Management: Immediate needle decompression (2nd intercostal space, midclavicular line) before imaging.

CXR Findings of Pneumothorax:
- Visceral pleural line: A thin white line (the visceral pleura) separated from the chest wall with NO lung markings beyond it. This is the hallmark finding.
- Absent lung markings peripherally: The area between the pleural line and chest wall contains only air (black, no vessels).
- Deep sulcus sign (supine CXR): On a supine film, air collects anteriorly and basally, causing an abnormally deep, lucent costophrenic angle. This is how you detect pneumothorax in a supine trauma patient.
- Increased lucency of the hemithorax.
- On a supine film, look for a sharp cardiac border and sharp anterior costophrenic angle.

Size Assessment:
- BTS Method: Measure the distance from the lung edge to the chest wall at the hilum. Greater than 2 cm = large pneumothorax.
- ACCP Method: Measure at the apex. Greater than 3 cm = large.
- CT provides the most accurate volume assessment.

Pitfalls and Mimics:
- Skin folds: These project OVER the lung markings (lung vessels visible beyond the line). The pleural line in true pneumothorax has NO markings beyond it.
- Bullae in emphysema: Can mimic loculated pneumothorax. CT helps differentiate.
- Artifact from clothing or bedding.
- Medial border of scapula can simulate a pneumothorax line.

CT vs. CXR:
CT is significantly more sensitive than CXR for detecting pneumothorax, especially small and anterior pneumothoraces. Occult pneumothorax (visible only on CT) is found in up to 10-15% of trauma patients with a normal CXR.

Management Principles:
- Small primary spontaneous: Observation, supplemental oxygen (accelerates resorption)
- Large or symptomatic: Aspiration or chest drain (intercostal tube)
- Tension: IMMEDIATE needle decompression followed by chest drain
- Recurrent: Consider pleurodesis or VATS (video-assisted thoracoscopic surgery) with blebectomy
"""
    },

    {
        "title": "Pleural Effusion — Imaging Characteristics and Differential Diagnosis",
        "source_type": "textbook",
        "source_attribution": "Based on StatPearls 'Pleural Effusion' and Light's Criteria",
        "content": """
Pleural Effusion — Comprehensive Imaging Guide

Definition: Pleural effusion is an abnormal accumulation of fluid in the pleural space between the visceral and parietal pleura.

Normal Pleural Fluid: The pleural space normally contains approximately 5-15 mL of fluid, which acts as a lubricant.

CXR Detection Thresholds:
- Erect PA film: ~200-300 mL needed to blunt the costophrenic angle (earliest sign)
- Lateral film: ~50-75 mL visible as blunting of the posterior costophrenic angle (more sensitive)
- Lateral decubitus film: Can detect as little as 50 mL (most sensitive plain film technique)
- Supine film: 500+ mL may be present with only subtle haziness (supine films are unreliable for effusion detection)

CXR Findings by Size:
- Small: Blunting of the costophrenic angle (meniscus sign)
- Moderate: Opacity extending up the lateral chest wall with a concave upper border (meniscus sign). Obscuration of the hemidiaphragm.
- Large: Opacification of most of the hemithorax. Contralateral mediastinal shift (if shift is TOWARD the effusion, suspect underlying collapse or mesothelioma).
- Massive: Complete hemithorax opacification ("white-out"). Must differentiate from complete lung collapse.

Ultrasound Features (gold standard for detection):
- Anechoic (black) space between parietal and visceral pleura
- "Quad sign" — four-sided echogenic structure (chest wall, pleura, lung, rib shadow)
- "Sinusoid sign" — respiratory variation in distance between lung and chest wall
- Can detect as little as 5-20 mL
- Reliably guides thoracentesis, reducing procedural complications

CT Features:
- Dependent, crescentic fluid density collection
- Attenuation values help characterize: simple transudative effusions measure 0-20 HU; complex/exudative/hemorrhagic effusions may measure 20-40+ HU
- Enhancement of the pleura (pleural thickening/enhancement) suggests exudative effusion or empyema
- "Split pleura sign" — enhancement of both visceral and parietal pleura with fluid between them — strongly suggests empyema

Types and Differentiation:

TRANSUDATIVE (systemic causes — the pleura is normal):
- Congestive heart failure (most common cause overall)
- Cirrhosis (hepatic hydrothorax)
- Nephrotic syndrome
- Hypoalbuminemia
- CXR: Bilateral, right more than left; associated cardiomegaly and pulmonary edema

EXUDATIVE (local pleural disease — inflammation/infection/malignancy):
- Pneumonia (parapneumonic effusion) — most common exudative cause
- Malignancy (lung, breast, lymphoma, mesothelioma)
- Pulmonary embolism
- Tuberculosis
- Autoimmune (rheumatoid arthritis, SLE)
- CXR: Often unilateral; may have associated consolidation, mass, or lymphadenopathy

HEMORRHAGIC (hemothorax):
- Trauma, malignancy, pulmonary infarction, aortic dissection
- Hematocrit of pleural fluid > 50% of blood hematocrit = hemothorax

Light's Criteria (for exudate):
An effusion is exudative if ANY ONE of the following is met:
1. Pleural protein / serum protein ratio > 0.5
2. Pleural LDH / serum LDH ratio > 0.6
3. Pleural LDH > 2/3 the upper limit of normal serum LDH

Special Types:
- Loculated effusion: Does not shift with position changes. Often post-inflammatory. May require CT or ultrasound-guided drainage.
- Subpulmonic effusion: Mimics elevated hemidiaphragm. Lateral decubitus film reveals the free fluid.
- Fissural pseudotumor ("phantom tumor"): Fluid trapped in a fissure, appears as a well-defined oval opacity. Disappears with diuresis in heart failure patients.
"""
    },

    # ──────────────────────────────────────────────────────────
    # CARDIAC AND VASCULAR IMAGING
    # ──────────────────────────────────────────────────────────

    {
        "title": "Pulmonary Edema — Cardiogenic vs Non-Cardiogenic",
        "source_type": "textbook",
        "source_attribution": "Based on StatPearls 'Pulmonary Edema' and Radiology Review Manual (Dahnert)",
        "content": """
Pulmonary Edema — Radiographic Differentiation

Definition: Pulmonary edema is excess fluid accumulation in the lung parenchyma and/or alveoli. Differentiation between cardiogenic and non-cardiogenic edema is critical as management differs fundamentally.

CARDIOGENIC PULMONARY EDEMA (Hydrostatic):
Cause: Elevated pulmonary capillary wedge pressure (>18 mmHg) due to left heart failure.

Progressive CXR Findings (by pulmonary capillary wedge pressure):
Stage 1 — Pulmonary Venous Hypertension (PCWP 12-18 mmHg):
  - Cephalization (upper lobe pulmonary venous distension): Normally, lower lobe vessels are larger than upper lobe due to gravity. When pressure rises, upper lobe vessels become equal to or larger than lower lobe vessels.
  - This is the EARLIEST sign of cardiogenic edema on CXR.

Stage 2 — Interstitial Edema (PCWP 18-25 mmHg):
  - Kerley B lines: Short (1-2 cm), horizontal lines perpendicular to the pleural surface at the lung bases. They represent thickened interlobular septa filled with fluid.
  - Peribronchial cuffing: Thickening around bronchi seen en face (bronchial wall thickening).
  - Perihilar haze: Blurring of the hilar structures.
  - Thickened interlobar fissures.

Stage 3 — Alveolar Edema (PCWP >25 mmHg):
  - Bilateral perihilar consolidation: Classic "bat-wing" or "butterfly" pattern — bilateral, symmetric, central opacities sparing the periphery.
  - Air bronchograms may be present within the consolidation.
  - Bilateral pleural effusions (right > left in 70% of cases).
  - Cardiomegaly (CTR > 0.5 on PA film).

Key Features of Cardiogenic Edema:
  - Central/perihilar distribution (gravitational, bat-wing)
  - Bilateral and relatively symmetric
  - Associated cardiomegaly
  - Bilateral pleural effusions
  - Cephalization of vessels
  - Kerley B lines
  - Rapid improvement with diuresis (can clear within hours)

NON-CARDIOGENIC PULMONARY EDEMA (ARDS / Permeability Edema):
Cause: Damage to the alveolar-capillary membrane (normal PCWP <18 mmHg).

Causes include: Sepsis, aspiration, pneumonia, trauma, pancreatitis, transfusion-related acute lung injury (TRALI), drug overdose, drowning, high altitude.

CXR Findings:
  - Bilateral diffuse opacities (ground-glass or consolidation)
  - PERIPHERAL distribution (opposite of cardiogenic — worse at the edges)
  - Normal heart size (no cardiomegaly)
  - NO cephalization of vessels
  - NO Kerley B lines
  - Pleural effusions absent or small
  - Air bronchograms within areas of consolidation
  - Does NOT respond to diuresis

Berlin Criteria for ARDS:
  - Timing: Within 1 week of known clinical insult or new/worsening respiratory symptoms
  - Imaging: Bilateral opacities NOT fully explained by effusions, atelectasis, or nodules
  - Origin: Respiratory failure NOT fully explained by cardiac failure or fluid overload
  - Severity by PaO2/FiO2 ratio: Mild (200-300), Moderate (100-200), Severe (<100)

Differential Quick Reference:
| Feature | Cardiogenic | Non-Cardiogenic (ARDS) |
|---------|------------|----------------------|
| Distribution | Central (bat-wing) | Peripheral |
| Heart size | Enlarged | Normal |
| Effusions | Yes (bilateral) | Absent or small |
| Cephalization | Yes | No |
| Kerley B lines | Yes | No |
| Onset | Gradual | Acute |
| Diuresis response | Yes (rapid) | No |
"""
    },

    # ──────────────────────────────────────────────────────────
    # CT IMAGING — CRITICAL DIAGNOSES
    # ──────────────────────────────────────────────────────────

    {
        "title": "Pulmonary Embolism — CT Pulmonary Angiography (CTPA) Findings",
        "source_type": "textbook",
        "source_attribution": "Based on StatPearls 'Pulmonary Embolism Imaging' and PIOPED II study guidelines",
        "content": """
Pulmonary Embolism (PE) — CT Pulmonary Angiography Diagnosis

Definition: Pulmonary embolism is the obstruction of pulmonary arteries by thrombus, most commonly originating from deep veins of the lower extremities (DVT). PE is a life-threatening emergency.

CTPA (CT Pulmonary Angiography) — Gold Standard Imaging:
Sensitivity: 83-100%. Specificity: 89-97%.
CTPA has replaced conventional pulmonary angiography as the gold standard diagnostic test.

Protocol:
- IV contrast bolus with tracking in the main pulmonary artery
- Timing: Scan triggered when contrast reaches the pulmonary arteries (bolus tracking at 100-120 HU)
- Slice thickness: 1-1.25 mm for subsegmental detection

CTPA Findings:

Direct Signs:
1. Intraluminal Filling Defect: The definitive sign. Low-attenuation (dark) thrombus within the contrast-enhanced (bright) pulmonary artery. This can be:
   - Central (polo mint sign/railway track sign): Thrombus surrounded by contrast on axial images
   - Eccentric: Thrombus adherent to one wall, forming an acute angle with the vessel wall
   - Occlusive: Complete obstruction of the vessel with absence of contrast beyond the thrombus

2. Saddle Embolus: Large thrombus straddling the bifurcation of the main pulmonary artery into right and left branches. Associated with hemodynamic instability and high mortality.

Indirect Signs (suggest PE even if thrombus is not directly visualized):
- Right ventricular dilation: RV/LV ratio > 1.0 (measured on axial images). This indicates right heart strain and is an independent predictor of mortality.
- Interventricular septum bowing toward the left ventricle (D-sign on short axis)
- Reflux of contrast into the IVC and hepatic veins (tricuspid regurgitation)
- Pulmonary infarction: Peripheral wedge-shaped consolidation (Hampton hump equivalent)
- Mosaic perfusion: Areas of different attenuation in the lung parenchyma due to heterogeneous blood flow
- Main pulmonary artery diameter > 29 mm (pulmonary hypertension)

CXR Findings in PE (often normal — CXR does NOT rule out PE):
- Hampton Hump: Peripheral, wedge-shaped opacity with base against the pleura (pulmonary infarction). Present in <20% of cases.
- Westermark Sign: Focal oligemia (decreased vascularity) distal to the embolus. Rare but specific.
- Fleischner Sign: Enlarged central pulmonary artery. Suggests massive PE.
- Pleural effusion: Small, unilateral. Present in 40-50% of PE cases.
- Normal CXR: The most common finding in PE. A normal CXR in a dyspneic patient should INCREASE suspicion for PE.

Acute vs Chronic PE on CT:
| Feature | Acute PE | Chronic PE |
|---------|---------|-----------|
| Filling defect | Central, occlusive | Eccentric, mural, organized |
| Vessel size | Normal or enlarged | May be reduced (web/band) |
| Angle with wall | Acute angle | Obtuse angle (adherent) |
| RV dilation | Acute dilation | Chronic RV hypertrophy |
| Other | No calcification | May calcify, webs/bands |

Wells Score for Clinical Probability:
- Clinical signs of DVT: +3
- PE more likely than alternative diagnosis: +3
- Heart rate > 100: +1.5
- Immobilization or surgery in past 4 weeks: +1.5
- Previous DVT/PE: +1.5
- Hemoptysis: +1
- Malignancy: +1
Score interpretation: Low (0-1), Moderate (2-6), High (≥7)

Follow-up: D-dimer is used to rule out PE in low-probability patients (high sensitivity, low specificity). CTPA is indicated for moderate-to-high probability or positive D-dimer.
"""
    },

    {
        "title": "Acute Stroke — CT and MRI Imaging Findings",
        "source_type": "textbook",
        "source_attribution": "Based on StatPearls 'Stroke Imaging' and AHA/ASA Stroke Imaging Guidelines",
        "content": """
Acute Stroke — Neuroimaging Diagnosis

Stroke is the sudden loss of brain function due to interruption of blood supply (ischemic — 87%) or rupture of blood vessels (hemorrhagic — 13%). Imaging is CRITICAL because treatment (thrombolysis/thrombectomy) is extremely time-sensitive.

"Time is brain" — approximately 1.9 million neurons die per minute during an ischemic stroke.

NON-CONTRAST CT (NCCT) — First-Line Imaging:
The primary role of NCCT in acute stroke is to EXCLUDE hemorrhage before thrombolysis, not to confirm ischemia (early ischemia is often invisible on CT).

CT Findings of ACUTE ISCHEMIC STROKE (by time):

Hyperacute Phase (0-6 hours):
- CT may be NORMAL in up to 60% of cases in the first 6 hours
- Hyperdense vessel sign (dense MCA sign): Hyperdense (bright) middle cerebral artery due to intraluminal thrombus. Sensitivity ~30% but highly specific.
- Dot sign: Hyperdense thrombus in the Sylvian fissure branch of MCA
- Subtle loss of grey-white matter differentiation
- Insular ribbon sign: Loss of definition of the insular cortex (lateral border of the insula becomes indistinct). This is one of the earliest CT signs.
- Sulcal effacement: Subtle swelling causing loss of normal sulcal spaces

Early Phase (6-24 hours):
- Progressive hypoattenuation (darkening) in the affected vascular territory
- Loss of grey-white matter differentiation becomes more obvious
- Sulcal effacement and early mass effect
- The hypodensity conforms to a specific vascular territory (MCA, ACA, PCA)

Subacute Phase (1-7 days):
- Well-defined hypoattenuation in the affected territory
- Progressive edema and mass effect (peaks at 3-5 days)
- Hemorrhagic transformation may occur (petechial or parenchymal)
- "Fogging effect" at 2-3 weeks: Temporary isodensity due to macrophage infiltration and neovascularization — the infarct appears to "disappear" then reappears

Chronic Phase (>2 months):
- Encephalomalacia: Well-defined hypodensity approaching CSF density
- Volume loss with ex-vacuo dilation of adjacent ventricle
- Chronic gliosis

CT ANGIOGRAPHY (CTA):
- Identifies the site of arterial occlusion (M1/M2 MCA, ICA, basilar)
- Evaluates collateral circulation
- Essential for thrombectomy planning (large vessel occlusion)
- CT Perfusion maps: CBF (cerebral blood flow), CBV (cerebral blood volume), MTT (mean transit time), Tmax — these define the ischemic core vs. salvageable penumbra

MRI FINDINGS (More sensitive than CT, especially early):

DWI (Diffusion-Weighted Imaging):
- THE MOST SENSITIVE SEQUENCE for acute ischemia (sensitivity >95% within 30 minutes)
- Acute infarction appears BRIGHT (hyperintense) on DWI with corresponding LOW signal on ADC map (restricted diffusion)
- This is because ischemia causes cytotoxic edema — cells swell, restricting water molecule movement

FLAIR (Fluid-Attenuated Inversion Recovery):
- Becomes positive after 4-6 hours
- DWI-FLAIR mismatch: Positive DWI but negative FLAIR suggests the stroke is less than 4.5 hours old — useful when onset time is unknown (wake-up stroke)

Gradient Echo (GRE) / SWI:
- Susceptibility-weighted sequences detect blood products (hemorrhage)
- "Blooming artifact" from hemorrhage appears as dark signal
- Detects hemorrhagic transformation
- Microbleeds appear as small dark foci — many microbleeds increase the risk of hemorrhagic transformation with thrombolysis

CT FINDINGS OF HEMORRHAGIC STROKE:
- Acute blood is HYPERDENSE (bright white, 50-70 HU) on NCCT — this is why CT is the first test in stroke
- Intraparenchymal hemorrhage (IPH): Hyperdense lesion within brain parenchyma
- Surrounding hypodense edema develops within hours
- "Spot sign" on CTA: Active contrast extravasation within the hematoma, predicts hematoma expansion and poor outcome
- Location helps determine etiology: Basal ganglia/thalamus = hypertensive; Lobar = amyloid angiopathy or mass; Multiple locations = coagulopathy

HEMORRHAGE TYPES:
- Epidural hematoma: Biconvex (lens-shaped), does NOT cross sutures, often temporal (middle meningeal artery). "Lucid interval" then rapid deterioration.
- Subdural hematoma: Crescent-shaped, crosses suture lines but NOT the midline falx. Acute = hyperdense, chronic = hypodense.
- Subarachnoid hemorrhage (SAH): Hyperdensity in sulci and cisterns. Most common cause: ruptured berry aneurysm. "Thunderclap headache."
"""
    },

    # ──────────────────────────────────────────────────────────
    # ABDOMINAL IMAGING
    # ──────────────────────────────────────────────────────────

    {
        "title": "Acute Abdomen — Imaging Approach and Key Findings",
        "source_type": "textbook",
        "source_attribution": "Based on StatPearls 'Acute Abdomen Imaging' and ACR Appropriateness Criteria",
        "content": """
Acute Abdomen — Systematic Imaging Approach

Definition: Acute abdomen refers to sudden onset of severe abdominal pain requiring urgent evaluation. Imaging plays a critical role in diagnosis and guiding management.

Imaging Modality Selection:

ABDOMINAL X-RAY (AXR):
Role: Limited but still useful for bowel obstruction and perforation detection.
Standard Views: Supine and erect abdomen; erect chest X-ray for free air.

Key AXR Findings:
1. Small Bowel Obstruction (SBO):
   - Dilated small bowel loops > 3 cm diameter
   - Valvulae conniventes (plicae circulares): Thin mucosal folds that cross the entire width of the bowel lumen — characteristic of small bowel
   - Multiple air-fluid levels on erect film (> 2.5 cm difference between levels in the same loop = "differential air-fluid levels")
   - Collapsed/decompressed distal bowel
   - "String of beads" sign: Small gas bubbles trapped between valvulae on erect film
   - Rule of 3s: Small bowel > 3 cm, Large bowel > 6 cm, Cecum > 9 cm = dilated

2. Large Bowel Obstruction (LBO):
   - Dilated large bowel > 6 cm (cecum > 9 cm = risk of perforation)
   - Haustra: Thick folds that do NOT cross the entire lumen width — identifies large bowel
   - Most common cause: Colorectal carcinoma (60%), followed by volvulus (20%)
   - Cecal diameter > 12 cm = imminent perforation risk

3. Pneumoperitoneum (Free Air):
   - Erect CXR: Air under the diaphragm (most sensitive plain film method)
   - Supine AXR: Rigler's sign (both sides of the bowel wall visible — double wall sign)
   - Football sign: Large pneumoperitoneum outlines the entire peritoneal cavity
   - Falciform ligament sign: Air outlines the falciform ligament
   - Most common cause: Perforated peptic ulcer, diverticulitis, or post-surgical

4. Volvulus:
   - Sigmoid volvulus: "Coffee bean sign" — large, dilated sigmoid loop arising from the pelvis, pointing toward the right upper quadrant. Absence of haustral markings. "Inverted U" shape.
   - Cecal volvulus: Dilated cecum displaced to the left upper quadrant. "Kidney bean" appearance.

CT ABDOMEN (with IV contrast) — Workhorse of Acute Abdomen:

Appendicitis:
- Distended appendix > 6 mm in diameter
- Appendiceal wall thickening and enhancement
- Periappendiceal fat stranding (the most sensitive CT sign)
- Appendicolith (calcified fecalith) — present in ~25% of cases
- Complications: Perforation (free fluid, abscess, extraluminal air), phlegmon

Bowel Obstruction (CT Features):
- Transition point: Abrupt change from dilated to collapsed bowel
- Small bowel feces sign: Particulate matter in dilated small bowel near transition point
- "Closed loop" obstruction: U-shaped dilated loop with converging mesentery — surgical emergency (risk of strangulation and bowel ischemia)
- Strangulation signs: Reduced/absent bowel wall enhancement, mesenteric haziness, hemorrhagic ascites, pneumatosis (air in bowel wall)

Diverticulitis:
- Pericolonic fat stranding centered on a diverticulum
- Thickened colonic wall
- Complications: Abscess, perforation, fistula
- Modified Hinchey Classification guides management

Renal/Ureteric Calculi:
- Non-contrast CT is the gold standard (sensitivity >95%)
- Direct visualization of calculus in the ureter
- Secondary signs: Hydroureter, hydronephrosis, perinephric stranding
- "Tissue rim sign": Soft tissue rim around the stone due to ureteral edema

ULTRASOUND — Key Applications:
- Right upper quadrant pain: Cholecystitis (gallstones, wall thickening > 3mm, pericholecystic fluid, positive Murphy sign)
- Right iliac fossa: Appendicitis (non-compressible appendix > 6mm, target sign)
- Pelvic pain in women: Ovarian torsion, ectopic pregnancy, tubo-ovarian abscess
- Aortic aneurysm screening and emergency diagnosis of rupture
"""
    },

    # ──────────────────────────────────────────────────────────
    # MUSCULOSKELETAL RADIOLOGY
    # ──────────────────────────────────────────────────────────

    {
        "title": "Fracture Assessment — Radiographic Principles and Commonly Missed Fractures",
        "source_type": "textbook",
        "source_attribution": "Based on StatPearls 'Fracture Assessment' and Emergency Radiology guidelines",
        "content": """
Fracture Assessment — Systematic Radiographic Approach

Fundamental Principles:
1. Always obtain at least TWO views at 90 degrees to each other (AP and lateral minimum)
2. Image the joint above and below the suspected fracture
3. Compare with the contralateral (opposite) side if uncertain, especially in children
4. "One fracture = look for a second" — associated injuries are common

Describing Fractures — Standard Terminology:

Location: Which bone, which part (proximal, mid-shaft, distal, intra-articular)
Type: Transverse, oblique, spiral, comminuted (>2 fragments), segmental, avulsion
Displacement: Undisplaced, minimally displaced, or describe direction and degree
Angulation: Direction and degree of the distal fragment relative to proximal
Open vs Closed: Open fracture if associated with overlying skin wound (clinical, not radiographic)
Involvement: Intra-articular (involves the joint surface) vs extra-articular

COMMONLY MISSED FRACTURES (HIGH-RISK for medicolegal claims):

1. Scaphoid Fracture:
   - Most commonly fractured carpal bone (70% of carpal fractures)
   - Mechanism: Fall on outstretched hand (FOOSH)
   - Initial X-ray is NORMAL in 15-20% of cases
   - Findings when visible: Lucent line through the scaphoid waist
   - If clinical suspicion (anatomical snuffbox tenderness) and normal X-ray: Treat as fracture, repeat imaging at 10-14 days, or obtain MRI (most sensitive)
   - Complication: Avascular necrosis (AVN) of the proximal pole due to retrograde blood supply

2. Neck of Femur (Hip Fracture):
   - Critical in elderly patients after falls
   - May be subtle or invisible on initial X-ray (especially undisplaced/impacted)
   - Look for: Disruption of Shenton's line, cortical irregularity, trabecular abnormality
   - If clinical suspicion and normal X-ray: MRI is the gold standard (100% sensitivity)
   - Garden Classification (for subcapital fractures): I-IV based on displacement and trabecular alignment
   - Complications: AVN (highest risk with displaced subcapital fractures), non-union

3. Radial Head Fracture:
   - Most common elbow fracture in adults
   - Mechanism: FOOSH with forearm pronation
   - Subtle fractures may be invisible — look for the "fat pad sign"
   - Fat Pad Signs: Anterior fat pad displacement ("sail sign") or posterior fat pad visibility = intra-articular fracture until proven otherwise, even if no fracture line is visible
   - Mason Classification: I (undisplaced), II (displaced >2mm), III (comminuted)

4. Lisfranc Fracture-Dislocation:
   - Tarsometatarsal joint injury — often missed on initial imaging
   - Key finding: Loss of alignment between the medial border of the 2nd metatarsal and the medial border of the middle cuneiform on AP view
   - "Fleck sign": Small avulsion fracture at the base of the 2nd metatarsal
   - Weight-bearing views are essential (injuries may reduce in non-weight-bearing)

5. Posterior Malleolus Fracture:
   - Often missed if only AP view is obtained — requires lateral view
   - Present in 7-44% of ankle fractures

Pediatric-Specific Considerations:
- Salter-Harris Classification (growth plate injuries): Types I-V
  - Type I: Through the physis only (X-ray may be normal, widened physis)
  - Type II: Through physis and metaphysis (most common — 75%)
  - Type III: Through physis and epiphysis (intra-articular)
  - Type IV: Through metaphysis, physis, and epiphysis
  - Type V: Crush injury to physis (retrospective diagnosis)
  - Mnemonic: SALTR — Separated, Above, Lower, Through, Rammed/Ruined
- Greenstick fracture: Incomplete fracture with intact cortex on one side (unique to children's flexible bones)
- Torus (buckle) fracture: Compression failure of cortex, seen as cortical bulging without a discrete fracture line
"""
    },

    # ──────────────────────────────────────────────────────────
    # IMAGING MODALITIES — How each technology works
    # ──────────────────────────────────────────────────────────

    {
        "title": "Imaging Modalities — How X-Ray, CT, MRI, and Ultrasound Work",
        "source_type": "guideline",
        "source_attribution": "Based on Fundamentals of Diagnostic Radiology (Brant & Helms) and ACR Practice Guidelines",
        "content": """
Imaging Modalities in Radiology — Principles and Applications

1. CONVENTIONAL RADIOGRAPHY (X-RAY):

Physics: X-rays are a form of electromagnetic radiation. They pass through the body and are absorbed differently by different tissues. Dense structures (bone) absorb more → appear WHITE. Air absorbs least → appears BLACK.

Five Basic Radiographic Densities (from most to least dense/white):
- Metal (foreign bodies, prostheses) → Brightest white
- Bone/Calcium → White
- Soft tissue/Fluid → Grey
- Fat → Dark grey
- Air/Gas → Black

Advantages: Fast, inexpensive, widely available, low radiation dose
Limitations: 2D representation of 3D structures (superimposition), limited soft tissue contrast, poor sensitivity for subtle pathology
Common Applications: Chest (pneumonia, effusion), MSK (fractures), Abdomen (obstruction)

Radiation Dose: Chest PA = 0.02 mSv (equivalent to ~2.5 days of background radiation)

2. COMPUTED TOMOGRAPHY (CT):

Physics: CT uses X-rays taken from multiple angles, processed by a computer to create cross-sectional images. Modern helical/spiral CT scans continuously as the patient moves through the gantry.

Hounsfield Units (HU) — The CT density scale:
- Air: -1000 HU
- Fat: -100 to -50 HU
- Water: 0 HU
- Soft tissue: +20 to +80 HU
- Acute blood: +50 to +70 HU
- Bone: +400 to +1000 HU
- Metal: +2000+ HU

Contrast Enhancement:
- IV contrast (iodinated): Enhances blood vessels, highlights vascular structures and hypervascular lesions. Arterial phase (25-30 sec), portal venous phase (60-70 sec), delayed phase (3-5 min).
- Oral contrast: Opacifies the GI tract, helpful for bowel pathology.
- Contraindications: Contrast allergy (premedicate or use alternative), renal insufficiency (risk of contrast-induced nephropathy), metformin use (hold 48 hours after contrast).

Advantages: Excellent spatial resolution, fast acquisition, multiplanar reformats
Limitations: Ionizing radiation (higher dose than X-ray), IV contrast risks
Radiation Dose: CT Chest = 7 mSv, CT Abdomen/Pelvis = 10-15 mSv

3. MAGNETIC RESONANCE IMAGING (MRI):

Physics: MRI uses strong magnetic fields and radiofrequency pulses to excite hydrogen protons in the body. Different tissues return to equilibrium at different rates, creating contrast. NO ionizing radiation.

Key Sequences:
- T1-weighted: Fat is BRIGHT, water/fluid is DARK. Best for anatomy. "T1 = anatomy"
- T2-weighted: Water/fluid is BRIGHT, fat is bright. Best for pathology. "T2 = pathology"
- FLAIR: T2-weighted with CSF signal suppressed. Excellent for brain lesions near ventricles.
- DWI: Detects restricted diffusion (acute stroke, abscess). Bright signal = restricted diffusion.
- Gadolinium contrast: Enhances vascular structures, tumors, inflammation. Contraindicated in severe renal failure (risk of nephrogenic systemic fibrosis).

Mnemonic for MRI:
- "Water is White on T2-Weighted" (WWWT2W)
- "Fat is Bright on T1" (both have 1 syllable: Fat, T1)

Advantages: No radiation, superior soft tissue contrast, multiplanar imaging
Limitations: Expensive, long scan times, claustrophobia, contraindicated with certain metallic implants (pacemakers, cochlear implants, metallic foreign bodies)
Common Applications: Brain, spine, musculoskeletal (ligaments, menisci), pelvic, cardiac

4. ULTRASOUND (US):

Physics: High-frequency sound waves (2-18 MHz) are transmitted into the body. They bounce back (echo) when they hit tissue boundaries. The returning echoes create a real-time image. Higher frequency = better resolution but less penetration.

Terminology:
- Hyperechoic: Brighter than surrounding tissue (reflects more sound) — e.g., gallstones, fat
- Hypoechoic: Darker than surrounding tissue — e.g., solid masses
- Anechoic: Black, no echoes — e.g., simple cysts, fluid collections
- Posterior acoustic shadowing: Dark shadow behind dense objects (gallstones, calcifications)
- Posterior acoustic enhancement: Increased echogenicity behind fluid-filled structures (cysts)

Advantages: No radiation, real-time, portable, inexpensive, excellent for guided procedures
Limitations: Operator-dependent, limited by body habitus (obesity), poor for air-containing and deep structures
Common Applications: OB/GYN, hepatobiliary, thyroid, vascular, MSK, eFAST in trauma
"""
    },

    # ──────────────────────────────────────────────────────────
    # RADIOLOGY REPORTING & STANDARDS
    # ──────────────────────────────────────────────────────────

    {
        "title": "Structured Radiology Reporting — Best Practices and Standards",
        "source_type": "guideline",
        "source_attribution": "Based on ACR Practice Guidelines, RSNA Reporting Initiative, and ESR Guidelines",
        "content": """
Structured Radiology Reporting — Standards and Best Practices

WHY STRUCTURED REPORTING?
Free-text reports are variable, often miss key findings, and are difficult to mine for data. Structured reporting improves:
- Completeness: Ensures all relevant findings are addressed
- Clarity: Standardized language reduces ambiguity
- Communication: Referring physicians find answers faster
- Research: Structured data enables quality improvement and AI training

STANDARD REPORT SECTIONS:

1. EXAMINATION: Type of study, body part, with/without contrast
   Example: "CT Chest with IV Contrast"

2. CLINICAL INDICATION: Why the study was ordered
   Example: "56-year-old male with persistent cough and hemoptysis. Evaluate for malignancy."

3. COMPARISON: Previous imaging available for comparison
   Example: "Comparison: CT Chest dated 01/15/2024. Chest X-ray dated 02/20/2024."

4. TECHNIQUE: How the study was performed
   Example: "Helical CT from the thoracic inlet to the adrenal glands after IV administration of 100 mL Omnipaque 350. Arterial and venous phase images obtained."

5. FINDINGS: Systematic, organized description of all observations
   - Organize by body system or anatomical structure
   - Describe normal findings briefly (confirms assessment)
   - Describe abnormal findings in detail: location, size, morphology, change from prior
   - Use measurements (in cm/mm) for any lesion being followed
   - Example:
     "LUNGS: 2.3 cm spiculated nodule in the right upper lobe (series 3, image 45), increased from 1.8 cm on prior CT (3 months ago). Additional 4 mm ground-glass nodule in the left lower lobe, unchanged."

6. IMPRESSION: Concise summary with differential diagnosis and recommendations
   - Most important findings first
   - Numbered list for multiple findings
   - Include follow-up recommendations with timeframe
   - Reference standardized classification systems where applicable
   - Example:
     "1. Growing spiculated right upper lobe nodule, highly suspicious for primary lung malignancy (Lung-RADS 4B). Recommend PET-CT and tissue sampling.
      2. Stable left lower lobe ground-glass nodule. Recommend continued CT surveillance per Fleischner guidelines."

STANDARDIZED CLASSIFICATION SYSTEMS:

BI-RADS (Breast Imaging):
  0 — Incomplete, needs additional imaging
  1 — Negative (normal)
  2 — Benign finding
  3 — Probably benign (<2% malignancy risk). 6-month follow-up recommended.
  4 — Suspicious (4A: low, 4B: moderate, 4C: high suspicion). Biopsy recommended.
  5 — Highly suggestive of malignancy (≥95% probability). Biopsy.
  6 — Known biopsy-proven malignancy.

Lung-RADS (Lung Cancer Screening CT):
  1 — Negative (no nodules, or definitely benign)
  2 — Benign appearance. Annual screening.
  3 — Probably benign. 6-month follow-up CT.
  4A — Suspicious. 3-month follow-up CT or PET-CT.
  4B/4X — Very suspicious. Tissue sampling.

LI-RADS (Liver Imaging):
  LR-1 — Definitely benign
  LR-2 — Probably benign
  LR-3 — Intermediate probability
  LR-4 — Probably HCC
  LR-5 — Definitely HCC
  LR-M — Probably or definitely malignant, not specific for HCC
  LR-TIV — Tumor in vein

CRITICAL RESULT COMMUNICATION:
ACR Practice Parameter requires direct verbal communication for critical/unexpected findings:
- Findings requiring immediate intervention (tension pneumothorax, aortic dissection)
- New malignancy
- Findings markedly discrepant from clinical expectation
- Document: Time of communication, name of recipient, method of communication
"""
    },

    {
        "title": "Contrast Media — Types, Reactions, and Safety Protocols",
        "source_type": "guideline",
        "source_attribution": "Based on ACR Manual on Contrast Media (Version 2024) and ACR-SPR Practice Parameters",
        "content": """
Contrast Media in Radiology — Safety and Protocols

TYPES OF CONTRAST MEDIA:

1. IODINATED CONTRAST (for CT and fluoroscopy):
   - Mechanism: Attenuates X-rays → appears bright on CT
   - Types: Ionic vs Non-ionic (non-ionic preferred — fewer reactions)
   - Osmolality: Low-osmolar (LOCM) or iso-osmolar (IOCM) preferred
   - Common agents: Iohexol (Omnipaque), Iopamidol (Isovue), Iodixanol (Visipaque)
   - Administration: IV (most common), oral, rectal

2. GADOLINIUM-BASED CONTRAST (for MRI):
   - Mechanism: Alters local magnetic field → shortens T1 relaxation time → bright on T1
   - Types: Linear vs Macrocyclic (macrocyclic preferred — more stable, less retention)
   - Common agents: Gadobutrol (Gadavist), Gadoterate (Dotarem)
   - Risk: Nephrogenic Systemic Fibrosis (NSF) in severe renal failure (GFR <30)

ADVERSE REACTIONS TO IODINATED CONTRAST:

Allergic-like Reactions (not true allergy — anaphylactoid):
Mild (no treatment usually needed):
  - Hives, pruritus, scattered urticaria
  - Nasal congestion, sneezing
  - Nausea, vomiting
  - Transient flushing, warmth

Moderate (requires treatment):
  - Diffuse urticaria
  - Facial or laryngeal edema
  - Bronchospasm (wheezing)
  - Tachycardia or bradycardia

Severe (life-threatening — code blue):
  - Anaphylaxis with cardiovascular collapse
  - Respiratory arrest
  - Seizures
  - Cardiac arrest

Treatment Protocol:
- Mild: Observation, diphenhydramine 25-50 mg IV/IM if needed
- Moderate bronchospasm: Albuterol inhaler, epinephrine 0.1-0.3 mg IM if severe
- Severe anaphylaxis: Epinephrine 0.3 mg IM (1:1000), IV fluids, call code team
- ALWAYS have a crash cart available in the CT suite

Risk Factors for Contrast Reactions:
- Previous contrast reaction (5x increased risk)
- Asthma (10x increased risk if active)
- Multiple allergies

Premedication Protocol (for high-risk patients):
- Prednisone 50 mg PO at 13 hours, 7 hours, and 1 hour before contrast
- Diphenhydramine 50 mg IV/IM/PO 1 hour before contrast
- Use non-ionic, low-osmolar contrast agent

CONTRAST-INDUCED NEPHROPATHY (CIN):
- Definition: Increase in serum creatinine by ≥0.5 mg/dL or 25% above baseline within 48-72 hours of contrast administration
- Risk factors: Pre-existing renal insufficiency (eGFR <30), diabetes, dehydration, concurrent nephrotoxic drugs, high contrast volume
- Prevention: IV hydration with normal saline before and after contrast, minimize contrast volume, hold metformin for 48 hours after (risk of lactic acidosis if AKI develops)
- ACR recommendation: Check eGFR within 30 days for outpatients; within 48 hours for inpatients with known risk factors

METFORMIN AND CONTRAST:
- If eGFR ≥ 30: Administer contrast, no need to hold metformin
- If eGFR < 30: Generally do NOT give IV iodinated contrast. If essential, hold metformin 48 hours post-contrast and recheck renal function before resuming.

CONTRAST EXTRAVASATION:
- Definition: Leakage of contrast into subcutaneous tissue at the injection site
- Incidence: 0.1-0.9% of IV injections
- Management: Elevate the limb, apply ice for first 60 minutes, monitor for compartment syndrome (rare but surgical emergency)
- Large volume extravasation (>100 mL) or signs of compartment syndrome → surgical consult
"""
    },

    # ──────────────────────────────────────────────────────────
    # ONCOLOGIC IMAGING
    # ──────────────────────────────────────────────────────────

    {
        "title": "Lung Nodules — Assessment, Classification, and Follow-Up (Fleischner Society)",
        "source_type": "guideline",
        "source_attribution": "Based on Fleischner Society 2017 Guidelines and ACR Lung-RADS v1.1",
        "content": """
Pulmonary Nodule Assessment — Evidence-Based Management

Definition: A pulmonary nodule is a rounded or irregular opacity in the lung, well- or poorly-marginated, measuring ≤ 3 cm. Opacities > 3 cm are classified as "masses" and are treated with higher suspicion for malignancy.

Prevalence: Incidental pulmonary nodules are found in up to 50% of CT scans. The vast majority (>95%) are benign.

CHARACTERIZATION:

Size (single most important predictor of malignancy):
  - < 6 mm: Very low risk (<1%). Usually no follow-up needed (in low-risk patients).
  - 6-8 mm: Low risk (0.5-2%). Follow-up CT recommended.
  - 8-15 mm: Intermediate risk (3-15%). Consider PET-CT, tissue sampling, or surveillance.
  - > 15 mm: High risk (>15%). Further workup essential (PET-CT, biopsy).

Morphology:
  - Solid nodule: Completely obscures underlying lung parenchyma. Most common type.
  - Ground-glass nodule (GGN): Hazy opacity that does NOT obscure underlying structures. Higher per-size malignancy risk than solid nodules, but tend to be indolent cancers.
  - Part-solid nodule: Has both ground-glass and solid components. HIGHEST malignancy risk of all nodule types. The solid component determines aggressiveness.

Margins:
  - Smooth, well-defined: More likely benign (granuloma, hamartoma)
  - Lobulated: Intermediate risk
  - Spiculated (corona radiata): Highly suspicious for malignancy — spiculations represent tumor spreading along lymphatics and interstitium
  - Irregular: Suspicious

Calcification Patterns:
  - BENIGN patterns: Central, diffuse, laminated (concentric), popcorn (hamartoma)
  - INDETERMINATE/SUSPICIOUS: Eccentric, stippled, amorphous

Location: Upper lobe nodules carry higher malignancy risk.

FLEISCHNER SOCIETY GUIDELINES (2017) — Incidental Solid Nodules:

Low-Risk Patient (minimal/no smoking history, no other risk factors):
  - < 6 mm: No routine follow-up needed
  - 6-8 mm: CT at 6-12 months, then consider CT at 18-24 months
  - > 8 mm: CT at 3 months, PET-CT, or tissue sampling

High-Risk Patient (smoking history, family history, upper lobe, spiculated):
  - < 6 mm: Optional CT at 12 months
  - 6-8 mm: CT at 6-12 months, then CT at 18-24 months
  - > 8 mm: CT at 3 months, PET-CT, or tissue sampling

Growth Assessment:
  - Volume doubling time (VDT): Malignant nodules typically double in 20-400 days
  - VDT < 20 days: Likely infectious/inflammatory
  - VDT 20-400 days: Suspicious for malignancy
  - VDT > 400 days: Likely benign (but not always — GGN cancers can be very slow)
  - A 26% increase in diameter = volume doubling (because V = 4/3 π r³)

PET-CT Role:
  - FDG-avid (SUV > 2.5): Suspicious for malignancy, but false positives occur with infection/inflammation
  - Non-avid: Generally reassuring, but false negatives occur with slow-growing tumors (GGN adenocarcinomas) and carcinoid tumors
  - Most useful for solid nodules > 8 mm

DIFFERENTIAL DIAGNOSIS OF A SOLITARY PULMONARY NODULE:
Benign: Granuloma (most common benign cause — histoplasmosis, TB), hamartoma (popcorn calcification + fat on CT), arteriovenous malformation, round atelectasis, intrapulmonary lymph node
Malignant: Primary lung cancer (adenocarcinoma most common), solitary metastasis (colon, renal, breast, melanoma, sarcoma), carcinoid tumor, lymphoma
"""
    },

    # ──────────────────────────────────────────────────────────
    # EMERGENCY RADIOLOGY
    # ──────────────────────────────────────────────────────────

    {
        "title": "Trauma Imaging — eFAST, CT Trauma Protocol, and Critical Injuries",
        "source_type": "guideline",
        "source_attribution": "Based on ATLS (Advanced Trauma Life Support) Guidelines and ACR Appropriateness Criteria for Trauma",
        "content": """
Trauma Imaging — Systematic Approach to the Injured Patient

IMAGING IN TRAUMA — HIERARCHY:

1. eFAST (Extended Focused Assessment with Sonography for Trauma):
   - Performed at the bedside within minutes of arrival
   - 4 standard views + 2 extended views:
     a) Right upper quadrant (Morison's pouch — hepatorenal recess): Most sensitive for free fluid
     b) Left upper quadrant (splenorenal recess): Free fluid around spleen
     c) Suprapubic (pelvic/bladder): Free fluid in the pelvis (rectovesical pouch in males, pouch of Douglas in females)
     d) Subxiphoid (pericardium): Pericardial effusion/tamponade
     e+f) Bilateral anterior chest: Pneumothorax (absent lung sliding = pneumothorax)
   - Positive eFAST = free intraperitoneal fluid (blood in trauma) → may require emergent laparotomy in unstable patients
   - Sensitivity for free fluid: ~85-96%. Limited for solid organ injury without significant hemorrhage.

2. TRAUMA CT (Whole-Body CT / Pan-Scan):
   - Performed in hemodynamically stable or stabilized patients
   - Protocol: Non-contrast head + CT Angiography (CTA) of neck + Contrast-enhanced chest/abdomen/pelvis
   - "Pan-scan" has become standard of care in major trauma (reduces missed injuries)

CRITICAL INJURIES BY REGION:

HEAD:
- Epidural hematoma: Biconvex, lens-shaped, hyperdense. Usually temporal (middle meningeal artery rupture). Doesn't cross sutures. Neurosurgical emergency if causing midline shift.
- Subdural hematoma: Crescent-shaped, crosses sutures but not midline. Acute = hyperdense. May require surgical evacuation if >10mm thick or >5mm midline shift.
- Traumatic subarachnoid hemorrhage: Hyperdensity in sulci/cisterns. More common in elderly and those on anticoagulants.
- Diffuse axonal injury (DAI): Often CT-negative initially. MRI with SWI shows multiple small hemorrhagic foci at grey-white junction, corpus callosum, and brainstem. Mechanism: Rotational acceleration/deceleration.
- Skull fractures: Linear (most common), depressed (if fragments depressed > full thickness of skull → surgical), basilar (look for fluid in mastoid air cells, sphenoid sinus).

SPINE:
- Cervical spine: CT is the primary screening tool in trauma (replaces plain films)
- Unstable fractures: Jefferson fracture (C1 burst), Hangman's fracture (C2 pars interarticularis), odontoid fractures (Type II most unstable), burst fractures, flexion-distraction injuries (Chance fracture)
- MRI: For suspected ligamentous injury, spinal cord injury (cord edema/hemorrhage), or disc herniation with neurological deficit
- SCIWORA: Spinal cord injury without radiographic abnormality — more common in children

CHEST (Trauma CT Findings):
- Aortic injury: Most commonly at the aortic isthmus (just distal to the left subclavian artery origin). CT findings: Intimal flap, pseudoaneurysm, periaortic hematoma, mediastinal hemorrhage. Sudden deceleration mechanism.
- Pneumothorax and hemothorax
- Pulmonary contusion: Ground-glass opacity or consolidation that does NOT conform to lobar/segmental anatomy. Appears within 6 hours, peaks at 24-48 hours, resolves within 3-10 days.
- Rib fractures: Count and document (≥3 consecutive rib fractures = flail chest risk)
- Diaphragmatic rupture: More common on the left (right protected by liver). Look for herniation of abdominal contents into thorax.

ABDOMEN (Trauma CT — Solid Organ Injury Grading):

Spleen (most commonly injured abdominal organ in blunt trauma):
  - Grade I: Subcapsular hematoma < 10% surface area, laceration < 1 cm depth
  - Grade II: Subcapsular hematoma 10-50%, laceration 1-3 cm
  - Grade III: Subcapsular hematoma > 50% or ruptured, laceration > 3 cm
  - Grade IV: Laceration involving segmental or hilar vessels with >25% devascularization
  - Grade V: Shattered spleen or hilar vascular injury with devascularized spleen
  - Management: Grades I-III in stable patients → conservative (observation)
  - Active contrast extravasation ("blush") → angiographic embolization or surgery

Liver (second most commonly injured organ):
  - Grading similar to spleen (I-VI based on hematoma size, laceration depth, and vascular injury)
  - "Sentinel clot" sign: Highest attenuation blood clot adjacent to the source of bleeding
  - Active extravasation on arterial phase CT → emergent intervention

Kidney:
  - Grade I-II: Contusion, small hematoma. Conservative management.
  - Grade III: Laceration > 1cm without collecting system involvement
  - Grade IV: Laceration into collecting system (urine leak), renal artery/vein injury
  - Grade V: Shattered kidney, renal pedicle avulsion (devascularized)
  - Delayed images (5-10 min) essential to evaluate for urine leak (contrast extravasation)
"""
    },

]
