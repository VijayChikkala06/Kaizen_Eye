# Target industry and quantified business case for Kaizen Eye (phone-based, fully offline, learn-from-good AI visual inspection), India MSME focus, as of September 2026

Conventions used in these notes:
- Evidence labels: **[PRIMARY/MEASURED]** = peer-reviewed study, government release, or OEM document; **[VENDOR CLAIM]** = supplier marketing or press quote; **[ESTIMATE]** = market-research or modeled figure; **[SECONDARY]** = blog or aggregator, or a figure seen only in a search-engine summary that I could not open. Treat SECONDARY items as leads to re-check before they go on a slide.
- Currency conversions use these RBI reference rates, taken from search results that cite CEIC: USD 1 = ₹95.7245 (12 Sep 2026); EUR 1 = ₹110.045 (17 Sep 2026); GBP 1 = ₹127.029 (24 Sep 2026) — [CEIC USD](https://www.ceicdata.com/en/india/foreign-exchange-rate-reserve-bank-of-india/foreign-exchange-rate-rbi-reference-rate-us-dollars), [CEIC GBP](https://www.ceicdata.com/en/india/foreign-exchange-rate-reserve-bank-of-india/foreign-exchange-rate-rbi-reference-rate-pound-sterling). The CEIC pages returned HTTP 403, so these rates come from search snippets only [SECONDARY]. As a cross-check, ACMA's own FY26 conversion (₹7.60 lakh crore = USD 85.9 bn) implies an average of about ₹88.5/USD ([Autocar Pro](https://www.autocarpro.in/news/acma-indian-auto-component-industry-grows-127-percent-to-inr-76-lakh-crore-in-fy26-133454)). If the September 2026 rate is wrong, the USD figures below move by about 8%.
- Arithmetic done by me is marked "(calc)".

## 1. How well does manual visual inspection perform? (miss rates, fatigue, time per part, consistency)

### Takeaway
The research literature is consistent: human visual inspectors miss about 20–30% of defects, and even highly qualified inspectors catch only about 80–85% while wrongly rejecting a large share of good parts (35% in one Sandia study). Accuracy falls 13–45% with time on task, mostly within the first 30 minutes. Different inspectors vary widely, and the same inspector reverses about 23% of decisions when shown the same parts again. These figures give the "X" baseline for reliability.

### Cited Findings
- **Headline error rate [PRIMARY, review].** See (2012), Sandia report SAND2012-8590, reviewed 212 documents from the 1950s onward. It states that "error rates of 20% to 30% are frequently quoted in the inspection literature across multiple types of inspection tasks" (citing Drury & Fox, 1975). It also says that "even under 100% inspection, not all of the defects will be detected" (Drury, Karwan & Vanderwarker, 1986). — [See 2012, OSTI](https://www.osti.gov/servlets/purl/1055636); full text read from the [UNT mirror PDF](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Lower bound on error [PRIMARY, review].** Swain & Guttmann (1983) estimated a minimum error rate of about 10⁻³ for simple accept/reject tasks. For complex tasks the rate "generally exceeds 1 in 1000". — [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Observed detection rates, See 2012 Table 4 [PRIMARY, review].** Defects detected by task: measuring dimensions 9%–64% (Lawshe & Tiffin 1945); surface defects on piston rings 67% (Hayes 1950); soldering defects 45%–100% (Jacobson 1952); acoustical tiles 76% (Carter 1957); magnetic-particle inspection of aircraft landing gear 57%–98% (Heida 1989); aircraft visual inspection 68% (Drury et al. 1997); subsea structures 53%; highway bridges 52%. The PDF text layout was garbled, so I matched rows to values by their order. Re-check against the PDF table before quoting. — [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Expert inspectors: hit rate and false rejects [PRIMARY, measured].** See (2015), *Human Factors*: 82 inspectors from the US Nuclear Security Enterprise inspected 140 precision parts for 8 defect types. They "correctly rejected 85% of defective items and incorrectly rejected 35% of acceptable parts". This hit rate was "not vastly superior to the industry average of 80%" and came at the cost of a high scrap rate. — [Sandia publication page](https://www.sandia.gov/research/publications/details/visual-inspection-reliability-for-precision-manufactured-parts-2015-12-01/); [DOI](https://doi.org/10.1177/0018720815602389)
- **Fatigue and vigilance over time [PRIMARY, review].**
  - Defect detection "could deteriorate up to 40% in 30 minutes" (Drury & Fox 1975).
  - Ten experienced inspectors checking **automotive rubber seals** (about 1 defect per 100 seals) showed a **27% drop in hits** from the first to the second 15-minute period (Fox 1977).
  - Field studies found a 13%–45% drop in hits with time on task (Drury & Watson 2002).
  - The decline is usually complete 20–35 minutes into a session, and at least half of it happens in the first 15 minutes (Teichner 1974).
  - The main recommendation is to limit inspection spells to 30 minutes or less.
  — [See 2012 §3.2.5](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Shift effects [PRIMARY, review].** Inspectors on night shifts are more likely to miss defects (Drury 1974). Late-night and early-morning shifts degraded baggage-screener performance (McCallum et al. 2005). — [See 2012 §3.2](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Consistency between and within inspectors [PRIMARY, review].**
  - Between inspectors: detection of solder defects "ranged from 43% to 100%, with no defect being detected by all inspectors".
  - Within the same inspector: when piston rings were inspected twice, "23% of the decisions were reversed" (McCornack 1961).
  - See calls large between- and within-inspector differences "perhaps the most consistent finding in inspection".
  — [See 2012 §3.3](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Line speed and pacing [PRIMARY, review].**
  - Doubling the normal inspection rate raised misses from **23% to 30%** (Schoonard, Gould & Miller 1973).
  - For integrated-circuit chips, performance levels off at about **120 seconds** per item, and "defects are usually detected quickly or not at all" (Schoonard & Gould 1973).
  - The best human pace for bottle inspection was about 200 bottles/min (46 ft/min). For sheet metal, speeds above 150 ft/min reduced accuracy.
  - Self-paced inspection generally beats externally paced inspection (Fox 1973).
  — [See 2012 §3.1.7](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Rare defects and multiple defect types [PRIMARY, review].**
  - Typical production defect rates are 1%–10%. Accuracy falls as the defect rate falls (Harris 1968: 16% down to 0.25%).
  - Sensitivity (d′) fell from 2.8 with one defect type to 1.6 with five types (Ainsworth 1982).
  - An inspector's accuracy on one defect type barely predicts accuracy on another (r = 0.20; McCornack 1961).
  — [See 2012 §3.1.1–3.1.2](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Two inspectors beat one [PRIMARY, review].** Having both inspectors check every item, and rejecting only when both reject, gave the best detectability (Drury, Karwan & Vanderwarker 1986). This supports positioning the app as a second inspector alongside the human. — [See 2012 §3.1.8](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Industrial PCB case [PRIMARY, review].** A printed-circuit assembly plant in Malaysia gave inspectors an average of 7.5 components per second. About 2.7% of boards shipped to customers were defective, costing roughly $300,000 a year (≈ ₹2.87 crore at today's rate, calc; historical figure). Ergonomic changes cut defects at the customer site by 2.5% within 12 weeks. — [See 2012 §3.2](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Simple visual aids help [PRIMARY, review].** Giving 27 machined-parts inspectors simple part drawings raised detection of objective defects by 42%, with no significant increase in time (Chaney & Teel 1967). — [See 2012 §3.1.9](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Textiles [SECONDARY; search snippet, article not opened].** "Only about 70% of defects can be detected by highly trained inspectors" in fabric inspection. The same search summary said fabric defects can cut selling price by 45%–65%. — [Sensors 2022, PMC9571054](https://pmc.ncbi.nlm.nih.gov/articles/PMC9571054/)
- **Time per part in automotive [VENDOR CLAIM, weak].** A vendor blog claims manual inspection takes about 38 s per part versus 2.4 s with AI. It also cites 45–60 s per camshaft (about 1.5 min with secondary inspection) and 87% detection by 12 inspectors checking 840 body panels per shift. No study design is given. — [iFactory blog](https://ifactoryapp.com/industries/automotive-manufacturing/ai-vs-manual-inspection-in-automotive-plants-speed-accuracy-and-cost)

### Inferences
- Credible human baselines for the slides:
  - **Miss rate 20–30%** (typical, from the literature).
  - **15% miss with 35% false rejects** (best case: expert, high-stakes inspectors).
  - **Up to 40% loss of detection within 30 minutes** (fatigue).
  - **23% of decisions reversed on re-inspection** (repeatability).
  - Use these as "X" values with the citations above, and say that they come from Western and aerospace/nuclear studies, not Indian MSMEs.
- The automotive rubber-seal result (27% fewer hits between the first and second 15 minutes) is the closest literature match to an Indian auto-component final-inspection table.
- The literature says defects are usually "detected quickly or not at all". The app's value is therefore consistency and fatigue immunity at the same pace, more than raw speed.

### Gaps
- I found no published Indian study measuring MSME inspectors' miss rates or time per part. The team should measure a local human baseline during the hackathon (see Section 7).
- I found no independent, peer-reviewed time-per-part figure for auto-component visual inspection. The 38 s/part figure is a vendor claim.
- The textile 70% figure was not read in full context.

## 2. What does manual inspection cost in India, and what is the cost of poor quality? (wages, COPQ, PPM expectations, ZED, warranty)

### Takeaway
A shop-floor inspector in India costs roughly ₹18,500–20,600 a month in wages alone (₹2.2–2.5 lakh a year per shift). The labour cost per inspection is therefore paise, not rupees. The real cost is quality failure:
- The ASQ rule of thumb puts quality-related costs at 15–20% of revenue.
- Indian OEM and Tier-1 supplier-quality manuals score suppliers on PPM bands where more than 500 PPM earns zero.
- A single non-conforming part in a sample can reject the whole lot (C=0 sampling), with 100% sorting charged to the supplier.

### Cited Findings
- **Haryana minimum wages from 1 Apr 2026 [SECONDARY; HR-portal summary of the state notification under the Code on Wages].**

  | Skill grade | Monthly | Daily |
  |---|---|---|
  | Unskilled | ₹15,220.71 | ₹585.41 |
  | Semi-skilled | ₹16,780.74 | ₹645.41 |
  | Skilled | ₹18,500.81 | ₹711.56 |
  | Highly skilled | ₹19,425.85 | ₹747.14 |

  These rates are about 35% higher than in July 2025, and the Zone A/B split was removed. This covers the Gurugram/Manesar/Faridabad auto cluster. — [Zimyo Haryana guide](https://www.zimyo.com/guides/minimum-wages/haryana/)
- **Market pay for QC inspectors [SECONDARY, salary aggregators, 2026].**
  - Indeed: about ₹20,603/month average — [Indeed](https://in.indeed.com/career/quality-control-inspector/salaries)
  - PayScale: ₹2,46,857/yr average, range ₹1.29–6.6 lakh — [PayScale](https://www.payscale.com/research/IN/Job=Quality_Control_Inspector/Salary)
  - SalaryExpert: ₹6,42,416/yr (₹309/h). This is a modelled estimate and looks high for shop-floor roles — [SalaryExpert](https://www.salaryexpert.com/salary/job/quality-inspector/india)
- **Cost of quality rule of thumb [SECONDARY summaries of ASQ].**
  - Quality-related costs reach **15–20% of sales revenue** at many manufacturers, and up to 40% of operations for poor performers.
  - World-class organisations keep cost of quality **below 5%**.
  - Poor-quality costs in manufacturing average about 15%, with a range of 5–35% depending on complexity.
  — [IISE "Measuring the Cost of Quality"](https://www.iise.org/details.aspx?id=22118); [Reliamag 2026](https://reliamag.com/articles/cost-of-poor-quality-scrap-rework-returns/); [Fabrico 2026](https://www.fabrico.io/blog/cost-of-poor-quality-copq-manufacturing-guide/)
- **Rejection baseline in an Indian small unit [PRIMARY, case study].** A Six Sigma DMAIC project at an Indian small-scale pressure die-casting unit cut rejection/rework from **15.50% to 4.47%** (sigma level 3.1 to 3.7, a 71.2% reduction) and saved ₹18,27,402. — [ResearchGate](https://www.researchgate.net/publication/290120371_Reducing_rejectionrework_in_pressure_die_casting_process_by_application_of_dmaic_methodology_of_six_sigma)
- **What an Indian OEM/Tier-1 demands of suppliers [PRIMARY, OEM document].** AVTEC (CK Birla group powertrain maker) Supplier Quality Manual, 1st edition, 01.08.2020:
  - Supplier score has a PPM component: 0 PPM = 25 points; 1–50 = 20; 51–100 = 15; 101–200 = 10; 201–500 = 5; **more than 500 PPM = 0**.
  - AVTEC "uses a C=0 sampling plan that rejects the entire lot when a single non-conforming part is found in the sample".
  - "100% sorting may be done as necessary at the supplier's expense".
  - Critical characteristics that do not meet capability "must be inspected 100%".
  - "Even the best receiving inspection program cannot detect all defective material".
  - The supplier is charged for any testing needed to judge non-conforming product.
  — [AVTEC SQM PDF](https://www.avtec.in/pdf/AVTEC%20Supplier%20Quality%20Manual.pdf)
- **Maruti's zero-PPM push [SECONDARY, dated 2013].** Maruti ran a "0 PPM" drive through Quality Circles, and about 22 Tier-II suppliers were reported to hold 0 PPM rejection. — [Business Standard, "Stretching the assembly line"](https://www.business-standard.com/article/management/stretching-the-assembly-line-113120800676_1.html)
- **ZED certification subsidy [SECONDARY; official page failed on a TLS error].**
  - Subsidy on certification cost: **80% for micro, 60% for small, 50% for medium** enterprises.
  - Joining reward of ₹10,000 after taking the ZED pledge.
  - Levels: Bronze, Silver, Gold.
  — [ZED portal, subsidy page](https://zed.msme.gov.in/subsidy-on-cost-of-certification); [SchemesInIndia summary](https://schemesinindia.in/central/zed-certification-msme); [Scheme guidelines PDF](https://www.dcmsme.gov.in/Guidelines_MSME%20Sustainable(ZED)%20Certification%20Scheme.pdf)

### Inferences (calc)
- **Wage cost per inspector per shift:**
  - At the Haryana skilled minimum: ₹18,500.81 × 12 = **₹2.22 lakh/yr** (≈ $2,319).
  - At the Indeed average: ₹20,603 × 12 = **₹2.47 lakh/yr** (≈ $2,583).
  - A station staffed on 2 shifts costs ₹4.4–4.9 lakh/yr; on 3 shifts, ₹6.7–7.4 lakh/yr.
  - These are wages only. Employer PF/ESI/bonus add-ons were not sourced here.
- **Labour cost per inspection** (208 working hours a month, so ₹89–99/h):
  - At 6 s/part (assumed), about **₹0.15–0.17 per part**.
  - At the vendor's 38 s/part, about **₹0.94–1.05 per part**.
  - Direct labour saving per part is small. The business case must rest on escapes avoided (PPM band, C=0 lot rejections, sorting charges, customer debits) and on consistency, not just on replacing headcount.
- **Illustrative escape arithmetic** (assumptions stated, not measured):
  - Incoming defect rate 2% (20,000 PPM; See 2012 says 1–10% is typical) with a 20–30% human miss rate means **4,000–6,000 PPM shipped**. That is far into AVTEC's ">500 PPM = 0 points" band.
  - Adding a second, independent screen with 95% recall (hypothetical, to be measured) would cut escapes to about 200–300 PPM, the 201–500 band.
  - The team should replace 95% with its measured recall.

### Gaps
- **Warranty and recall costs for Indian auto-component MSMEs:** not found.
- **COPQ as a % of revenue for Indian MSMEs specifically:** not found. One search summary attributed "rejection rates of MSMEs more than 3% can easily eat up the operating margin" to QCI, but I could not trace the original.
- **Maruti-specific PPM targets:** a search summary claimed about 50 PPM for safety-critical parts and 200–500 PPM for functional parts. I could not trace this to a primary Maruti document, so treat it as unverified.
- **Other state minimum wages:** Tamil Nadu (from 1 Apr 2026) and Maharashtra (from 1 Jul 2026) skilled rates were not retrieved.
- **ZED:** current counts of certified MSMEs, and whether ZED funds equipment (vs only certification and handholding), were not confirmed. A search summary cited targets of 54,000 certifications by 31 Mar 2027 and 2.5 lakh by 31 Mar 2028, but these are unverified.

## 3. What do commercial machine-vision systems cost, how long do they take to set up, and why don't MSMEs adopt them?

### Takeaway
The cheapest "learn from good parts" industrial system, Siemens Inspekto S70, was priced at €10,000 (≈ ₹11 lakh) in 2019. It still needs 20–30 good samples and 30–60 minutes to install. AI vision sensors sell in India for about ₹1.65–5 lakh per camera before lighting, fixtures and integration. Standard vision systems cost £5k–15k (≈ ₹6.4–19 lakh), and custom systems £15k–100k+. The main barriers named by Indian research are capital cost, setup, small batch sizes, and a shortage of technicians.

### Cited Findings
- **Siemens Inspekto S70 [VENDOR CLAIM, 2019].**
  - "Carries a price tag of only €10,000" (≈ ₹11.0 lakh ≈ $11,496, calc).
  - "Can be installed in 30 to 60 minutes".
  - One automotive plant reported direct savings of €468,336 a year at one location (≈ ₹5.15 crore, calc).
  — [ManufacturingTomorrow, 10 Apr 2019](https://www.manufacturingtomorrow.com/news/2019/04/10/inspekto-%E2%80%94-the-gold-mine-of-productivity/13232/)
- **Inspekto cost relative to traditional systems [VENDOR CLAIM, 2018].** The S70 "can be installed at ten per cent of the cost of a traditional machine vision solution". — [Quality Magazine, Dec 2018](https://www.qualitymag.com/articles/95159-inspekto-s70-autonomous-machine-vision-system)
- **Inspekto Gen 2 setup effort [VENDOR CLAIM, 2021; search-summary level].** Installs in as little as 45 minutes. Setup needs an average of 20–30 good samples, no defective samples, and no labelling. — [RoboticsTomorrow, 4 May 2021](https://www.roboticstomorrow.com/news/2021/05/04/inspekto-launches-the-second-generation-of-autonomous-machine-vision-systems/16776/); [Metrology News](https://metrology.news/autonomous-machine-vision-inspection-now-easier-and-more-powerful/)
- **AI vision-sensor prices in India [SECONDARY, IndiaMART dealer listings, 2025–26].**
  - Keyence IV3 (AI vision sensor): ₹1,65,000 (≈ $1,724) — [IndiaMART](https://www.indiamart.com/proddetail/keyence-vision-sensor-with-built-in-ai-iv3-series-2854380953255.html)
  - Keyence IV4: ₹4,99,999 (≈ $5,223) — [IndiaMART](https://www.indiamart.com/proddetail/keyence-iv4-vision-sensor-camera-with-built-in-ai-2857001721891.html)
  - Cognex In-Sight SnAPP: ₹1,80,000 per piece (≈ $1,880) — [IndiaMART](https://www.indiamart.com/proddetail/cognex-insight-snap-vision-sensor-2852699267873.html)
- **Cognex deep-learning camera, used [SECONDARY].** A used Cognex In-Sight D900 (ISD905M) was listed at $2,999.99 (≈ ₹2.87 lakh). New list prices are not published. — [eBay listing](https://www.ebay.com/itm/277471062774)
- **UK integrator price guide, 2026 [SECONDARY, vendor price guide].**

  | Item | GBP | INR (calc) |
  |---|---|---|
  | Standard vision systems | £5,000–15,000 | ₹6.35–19.05 lakh |
  | Custom systems | £15,000–100,000+ | ₹19 lakh–₹1.27 crore+ |
  | Deep-learning/AI development | £5,000–30,000+ | — |
  | Lighting | £50–3,000+ | — |
  | Lenses | £80–8,000+ | — |
  | Software | £1,000–15,000+ | — |
  | Service contracts | from £1,500/yr | — |

  — [Clearview Imaging](https://clearview-imaging.com/pages/how-much-does-machine-vision-actually-cost)
- **General system cost ranges [SECONDARY, search summary].** Systems range from $5,000 for a basic single-point setup to over $100,000 for multi-camera systems. Installation and integration typically cost $5,000–15,000. — [Opsio](https://opsiocloud.com/knowledge-base/how-much-does-a-vision-inspection-system-cost/)
- **Cloud deep-learning pricing model [PRIMARY, vendor docs].** LandingLens bills in credits: 1 credit per image trained and 1 credit per image inference. The Free plan gives 1,000 credits a month (no rollover) and 3 users. Enterprise pricing is "contact sales". — [LandingLens plans](https://landinglens.docs.landing.ai/plans)
- **Barriers to automation for MSMEs [PRIMARY, think-tank field study, 13 Apr 2026].**
  - About 80% of auto-component firms in the Delhi–Haryana cluster are Tier-2/Tier-3 MSMEs.
  - Barriers listed: capital expenditure limits; setup times and space; lack of scale, with **small batch orders of 5,000–10,000 units**; shortage of trained technicians; limited working capital.
  - Contract labour and absenteeism shape automation decisions.
  — [CSEP, "Wheels of Change"](https://csep.org/discussion-note/wheels-of-change-automation-in-indias-automotive-sector/)
- **MSME capability gap [PRIMARY survey via trade press, 3 Sep 2026].** Vector Consulting white paper:
  - 95% of industry leaders believe MSMEs are not investing fast enough in future capabilities.
  - Only 14% of MSMEs have systems-integration or product-development capability.
  - MSME auto-component turnover is estimated at ₹2.4–2.9 lakh crore.
  — [Autocar Professional](https://www.autocarpro.in/news/auto-component-msmes-face-capability-gap-rs-39000cr-tied-up-134505)
- **The idea itself is not new [SECONDARY, idea page, 27 Apr 2026].** "NetraQC", a *proposed* (not built) phone-camera AI inspection product for Indian small factories, suggests pricing of ₹999–2,999/month. It cites existing systems at $3,000–100,000+. — [StartupBasket](https://startupbasket.ai/ideas/netraqc-phone-ai-factory-inspection/)

### Inferences (calc)
- **Hardware cost per station, X vs Y.** Y = iQOO 15 at ₹72,999 MRP ([91mobiles](https://www.91mobiles.com/hub/iqoo-15-launched-in-india-price-availability/)). Stand and lighting costs were not sourced.

  | Incumbent (X) | Price | Multiple of phone cost |
  |---|---|---|
  | Keyence IV3 | ₹1.65 lakh | 2.3× |
  | Cognex SnAPP | ₹1.80 lakh | 2.5× |
  | Keyence IV4 | ₹5.0 lakh | 6.8× |
  | Inspekto S70 | ≈ ₹11.0 lakh | 15.1× |
  | Standard vision system | ₹6.35–19.05 lakh | 8.7–26.1× |
  | Custom system | ₹19 lakh–₹1.27 crore | 26–174× |

  The incumbent prices mostly exclude lighting and integration. The cheaper Snapdragon 8 Gen 5 phone, iQOO 15R at ₹44,999 ([BusinessToday](https://www.businesstoday.in/technology/news/story/iqoo-15r-launched-in-india-with-snapdragon-8-gen-5-soc-price-starts-at-rs-44999-517744-2026-02-24)), is a possible lower-cost Y, but only if the model is benchmarked on that chip.
- **Station cost in months of one skilled worker's wage** (Haryana minimum ₹18,501/month):

  | Station | Months of wage |
  |---|---|
  | Phone (iQOO 15) | ≈ 3.9 |
  | Keyence IV3 | ≈ 8.9 |
  | Keyence IV4 | ≈ 27 |
  | Inspekto S70 | ≈ 59.5 (about 5 years) |

  This is a crisp, judge-friendly framing of affordability.
- **Setup time.** The "learn only from good parts" idea already exists in industrial products (Inspekto: 20–30 good samples, 30–60 min). Kaizen Eye's novelty therefore has to be framed on four points:
  1. Runs on a phone the MSME may already own, fully offline on the NPU.
  2. Setup measured in minutes, from a short video.
  3. Live feed with an audible alert.
  4. About 1/15 the price of the cheapest learn-from-good system.
  The idea of phone-based inspection for MSMEs is also circulating at concept level (NetraQC). Judges may see it, so novelty must rest on measured execution.
- **Cloud credit pricing works against a high-volume MSME line.** At 1 credit per inference, a line inspecting 10,000 parts a day would need 10,000 credits a day. Fully offline, on-device inference has zero marginal cost and no dependence on connectivity or data sharing.

### Gaps
- No official list prices for Keyence, Cognex or Siemens were found. IndiaMART prices are dealer asks, and Inspekto's €10,000 is a 2019 figure.
- No sourced typical integration/commissioning time (days or weeks) for conventional rule-based or deep-learning vision projects in India.
- The LandingLens paid-tier price was not confirmed from an official page. Third-party sites mention "from $33/month".

## 4. Which sector should Kaizen Eye target? (candidate comparison, recommendation, and 1–2 backups)

### Takeaway
**Recommended primary target: Tier-2/Tier-3 MSME auto-component makers**, at the final inspection or pre-dispatch sorting table for small-to-medium machined, forged, cast, stamped, fastener and rubber parts. This sector has:
- the largest MSME-heavy manufacturing value pool (industry ₹7.60 lakh crore in FY26; MSME turnover ₹2.4–2.9 lakh crore);
- contractually enforced quality metrics (PPM bands, C=0 lot rejection, sorting at supplier cost);
- low automation because of capital cost and small batches;
- a 2026 OEM push toward AI visual inspection (Maruti).

**Backup 1: electronics/PCB assembly MSMEs** (fast-growing, well-documented human solder-inspection unreliability, free commercial-licence PCB benchmark data).
**Backup 2: MSME pharma, at-line or offline packaging checks** (regulatory pressure from revised Schedule M), with the caveat that inline blister lines are too fast for a phone.

### Cited Findings
- **Auto components, size [PRIMARY, ACMA via trade press, 7 Jul 2026]:**
  - FY26 turnover ₹7.60 lakh crore (USD 85.9 bn), up 12.7%.
  - OEM supplies ₹6,62,893 crore (+16.3%); aftermarket ₹1,08,453 crore (+9%).
  - Exports USD 24 bn (+5%); imports USD 25.4 bn.
  - 17% CAGR from FY21 to FY26.
  — [Autocar Professional](https://www.autocarpro.in/news/acma-indian-auto-component-industry-grows-127-percent-to-inr-76-lakh-crore-in-fy26-133454)
- **Auto components, MSME structure [PRIMARY]:**
  - MSMEs are about 80% of India's auto-component manufacturers, with turnover of ₹2.4–2.9 lakh crore — [Autocar Pro/Vector, 3 Sep 2026](https://www.autocarpro.in/news/auto-component-msmes-face-capability-gap-rs-39000cr-tied-up-134505)
  - The industry is 2.3% of GDP (FY2025) with 1.5 million direct workers; about 80% of Delhi–Haryana cluster firms are Tier-2/3 MSMEs; typical small batches are 5,000–10,000 units — [CSEP, 13 Apr 2026](https://csep.org/discussion-note/wheels-of-change-automation-in-indias-automotive-sector/)
- **Auto components, OEM pull toward AI inspection [PRIMARY, 30 Jan 2026].** Maruti Suzuki onboarded 5 startups from its incubation programme:
  - AugurAI: "AI-based visual inspection for defect identification in complex components".
  - Aatral: "AI-assisted inspection and 3D digital validation to eliminate visual defects, enabling suppliers to manufacture zero-defect components".
  - Indus Vision: AI visual inspection of finished vehicles.
  — [Maruti Suzuki press release](https://www.marutisuzuki.com/corporate/media/press-releases/2026/january/maruti-suzuki-onboards-5-more-startups-to-scale-new-age-technologies-across-business-areas)
- **Auto components, what suppliers face [PRIMARY]:** PPM scoring bands (more than 500 PPM = 0 points); C=0 lot rejection; 100% sorting at the supplier's expense; 100% inspection of critical characteristics that lack process capability. — [AVTEC SQM](https://www.avtec.in/pdf/AVTEC%20Supplier%20Quality%20Manual.pdf)
- **Auto components, rejection and defect evidence [PRIMARY]:**
  - An Indian small-scale die-casting unit had 15.50% rejection/rework before its Six Sigma project — [ResearchGate](https://www.researchgate.net/publication/290120371_Reducing_rejectionrework_in_pressure_die_casting_process_by_application_of_dmaic_methodology_of_six_sigma)
  - Inspectors of automotive rubber seals scored 27% fewer hits in the second 15-minute period than in the first (Fox 1977) — [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Electronics, size [PRIMARY, MeitY/PIB via search summary]:**
  - Electronics production estimated at ₹13.11 lakh crore in FY26, up from ₹1.90 lakh crore in FY15.
  - Mobile phones ₹6.27 lakh crore.
  - Electronics exports ₹4.24 lakh crore, or $47.96 bn in FY26.
  - About 25 lakh jobs created over the decade.
  — [PIB](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2284808&reg=48&lang=1); [IBEF](https://www.ibef.org/news/india-s-electronics-exports-surge-11-fold-to-us-47-98-billion-women-s-workforce-nears-30)
- **Electronics, human inspection unreliability [PRIMARY, review]:**
  - Solder-defect detection ranged from 43% to 100% across inspectors (McCornack 1961); Jacobson (1952) reported 45–100%.
  - In the Malaysian PCB plant, inspectors had 7.5 components per second, 2.7% of boards shipped defective, and about $300k a year was lost.
  — [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)
- **Electronics, ready-made proof data [PRIMARY].** VisA contains 10,821 images (9,621 normal, 1,200 anomalous) in 12 subsets. These include **PCB1–PCB4**, capsules, candles, macaroni, cashew, chewing gum and fryum. Licence is **CC BY 4.0 (commercial use allowed)** (ECCV 2022). — [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)
- **Pharma, pressure and size [PRIMARY/SECONDARY, news]:**
  - About 8,500 of about 10,500 pharma units are MSMEs.
  - The revised Schedule M deadline for MSMEs was extended to 31 Dec 2025, so all units are expected to comply from 1 Jan 2026.
  - In Himachal, only about 11% of firms reportedly applied for the extension.
  — [Business Standard](https://www.business-standard.com/health/small-pharma-companies-get-1-year-breather-to-implement-schedule-m-125021201111_1.html); [DrugsControl](https://drugscontrol.org/news-detail.php?newsid=41648); [The Tribune](https://www.tribuneindia.com/news/himachal/pharma-msmes-seek-2-yr-extension-to-meet-revised-schedule-m-norms/)
- **Pharma, line speeds [SECONDARY, search summary of trade articles].** Inline blister vision systems inspect about 300 packs/min (Lynx Spectra HR) up to 800 packs/min. — [Vision Systems Design](https://www.vision-systems.com/factory/article/16738302/packaging-and-production-embedded-vision-system-performs-blister-pack-inspection); [Assembly Magazine](https://www.assemblymag.com/articles/93773-vision-system-performs-flawless-blister-pack-inspection)
- **Ceramics/tiles (Morbi) [SECONDARY].** About 670 registered factories (Morbi Ceramic Manufacturers Association), or 800–1,200 units in broader counts. Turnover is estimated at ₹65,000–75,000 crore (2026). — [Republic World, May 2026](https://www.republicworld.com/initiatives/how-morbi-became-one-of-the-world-s-most-influential-ceramic-manufacturing-hubs-2026-05-27-125924); [Morbi Ceramic Manufacturers Association](https://ceramicassociation.com/aboutus)
- **Textiles [SECONDARY].** About 70% defect detection by trained fabric inspectors. — [PMC9571054](https://pmc.ncbi.nlm.nih.gov/articles/PMC9571054/)

### Inferences
Sector scoring, my judgement based on the findings above:

| Sector | Pain intensity | Phone-camera fit | Willingness to pay | Proof data for a hackathon | Verdict |
|---|---|---|---|---|---|
| **Auto components (Tier-2/3 MSME)** | Very high: PPM bands, C=0 lot rejection, sorting at supplier cost, OEM zero-defect push | High: small/medium parts at a manual station, self-paced at human speed, appearance defects | Medium: capex-constrained, but a ₹73k phone is below one IV3 camera | Real parts easy to source (bolts, nuts, washers, bearings, small castings); MVTec AD object categories | **Primary** |
| Electronics/PCBA and EMS MSMEs | High: fast growth; solder inspection highly variable between inspectors | Medium: small defects need macro or close focus and controlled light | Medium | VisA PCB1–4 (commercial-OK licence); spare PCBs easy to source | **Backup 1** |
| Pharma packaging (MSME units) | High: regulatory (revised Schedule M from 2026) | Low inline (300–800 blisters/min); OK for at-line sampling, line clearance, label and carton checks | High, but validation (GMP/computer-system validation) slows adoption | Capsules in VisA | **Backup 2** (at-line only) |
| Ceramics/tiles (Morbi) | Medium–high: grading affects price | Medium: large flat parts on fast conveyors | Low–medium: cluster under cost pressure in 2026 | — | Not recommended for the demo |
| Textiles/garments | High: about 30% of defects missed | Low for fabric rolls (wide, continuous web); OK for garment checking | Low | — | Not recommended |
| Food grading, jewellery, handicrafts | Variable | Natural variation makes "learn good" harder; jewellery is reflective | Low to high | — | Not recommended (no data gathered) |

- **Why auto components wins for Kaizen Eye's specific design** (few-shot learn-good, offline, beep):
  - Final inspection at MSMEs is typically a self-paced manual station, the setting where the literature shows fatigue-driven misses.
  - Parts are repetitive and rigid, so "good" appearance is stable, which suits anomaly detection.
  - Defects such as dents, scratches, burrs, missing threads or operations, cracks, blowholes and rust are visible appearance anomalies.
  - The customer's PPM band gives a direct money metric.
  - Batch sizes of 5,000–10,000 units and frequent changeovers favour minutes-level re-teaching over integrator-led projects.
  - This defect list is my inference; it is not from a source. Validate it with a local MSME.
- **Best narrative for judges:** "A Tier-2 supplier to an OEM that scores >500 PPM as zero and rejects whole lots on a single defect."

### Gaps
- **No verified count of auto-component MSMEs.** An often-repeated "~700 organised / ~10,000 unorganised players" figure appears to date from about 2015, and I could not trace it to a primary source.
- **No sourced line speeds or cycle times for Tier-2/3 final inspection.**
- **No sourced defect-type frequency data for Indian auto-component MSMEs.**
- **MVTec AD category names not captured.** The fetched page did not list the 15 categories. Background knowledge says they include screw and metal nut, but this needs verifying on the MVTec download page.
- **No data collected for FMCG packaging, food grading, jewellery or handicrafts.**

## 5. How large is the market, and which government schemes could fund adoption?

### Takeaway
Market-research estimates (which are not measurements) value global machine vision at about $15.8 bn in 2025, growing 8.3% a year to $23.6 bn by 2030. India's machine-vision systems market is about $0.63 bn in 2025 (≈ ₹6,000 crore). India's AI-vision segment is projected to grow about 27% a year. The addressable base is huge: 7.83 crore Udyam registrations by Feb 2026. Government schemes mainly subsidise certification and consultancy (ZED, LEAN) and demonstration centres (SAMARTH), not equipment purchases.

### Cited Findings
- **Global machine vision [ESTIMATE].** $15.83 bn (2025) to $23.63 bn (2030), CAGR 8.3% (report dated May 2025). Quality assurance and inspection is the largest application. Asia-Pacific is the fastest-growing region (9.2% CAGR), and India shows "the highest CAGR". — [MarketsandMarkets](https://www.marketsandmarkets.com/Market-Reports/industrial-machine-vision-market-234246734.html)
- **India machine-vision systems [ESTIMATE].** USD 626.55 m (2025) to USD 1,098.65 m by 2034, CAGR 5.57% (2026–2034). That is ≈ ₹5,998 crore rising to ≈ ₹10,517 crore (calc). — [IMARC](https://www.imarcgroup.com/india-machine-vision-systems-market)
- **India AI vision [ESTIMATE, search summary].** CAGR of 27.40% to 2032, with surveillance and industrial inspection the largest shares. — [MarketsandMarkets ResearchInsight](https://www.marketsandmarkets.com/ResearchInsight/india-ai-vision-market.asp)
- **Udyam registrations [PRIMARY, PIB via search summary].**
  - 7.83 crore enterprises registered on the Udyam Registration Portal plus the Udyam Assist Platform, as of 28 Feb 2026, up from 0.79 crore in FY22.
  - More than 7.9 crore by March 2026: 4.72 crore on the Udyam portal and 3.21 crore on the Assist Platform.
  — [PIB PRID 2246892](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2246892&reg=3&lang=1); [IBEF](https://www.ibef.org/news/over-7-83-crore-enterprises-registered-on-udyam-platforms-indicating-strong-msme-formalisation-growth)
- **MSME definition changed [PRIMARY, PIB via search summary].** From 1 Apr 2025, investment limits were raised 2.5× and turnover limits 2×. — [PIB Year-End Review 2025](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2209712&reg=3&lang=1)
- **ZED [SECONDARY].** Certification subsidy of 80%/60%/50% for micro/small/medium enterprises, plus a ₹10,000 joining reward. — [ZED portal](https://zed.msme.gov.in/subsidy-on-cost-of-certification); [SchemesInIndia](https://schemesinindia.in/central/zed-certification-msme)
- **MSME Competitive (LEAN) Scheme [SECONDARY].** The government contributes **90% of the implementation cost** for handholding and consultancy fees, with an extra 5% for SFURTI clusters, women/SC/ST-owned units and the North-East. Tools covered include 5S, Kaizen, Kanban, visual workplace and **poka-yoke**, with Basic, Intermediate and Advanced levels. — [Drishti IAS](https://www.drishtiias.com/daily-updates/daily-news-analysis/msme-competitive-lean-scheme); [QCI NDIE LEAN](https://ndie.qcin.org/programme/lean/)
- **SAMARTH Udyog Bharat 4.0 [PRIMARY, PIB via search summary].** "No financial assistance is given to any industry including MSME for adopting Industry 4.0 enabled technologies" under SAMARTH centres. It runs demonstration centres, reportedly CMTI Bengaluru, C4i4 Pune, IITD-AIA Delhi and CSIR-CMERI Durgapur. — [PIB PRID 2084143](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2084143&reg=48&lang=2); [Ministry of Heavy Industries](https://heavyindustries.gov.in/samarth-udyog-bharat-40)
- **Digital MSME [PRIMARY, guidelines].** Promotes ICT and cloud adoption by MSMEs, as a component of the CLCS-TUS scheme. — [DC-MSME guidelines PDF](https://www.dcmsme.gov.in/schemes/DigitalMSME-Guideline-CLCS-TUS-2019-2020.pdf)

### Inferences
- **Market numbers.** Present the MarketsandMarkets and IMARC figures as estimates, citing the firm and the date. Do not add them together. The strongest "market" argument for judges is bottom-up, not top-down: auto-component MSMEs (about 80% of manufacturers, ₹2.4–2.9 lakh crore turnover) × inspection stations per plant × ₹73k per station.
- **Policy fit.** The pitch that best fits policy is "a ₹73k poka-yoke/inspection aid that helps an MSME move up ZED levels and meet LEAN poka-yoke goals". Don't claim the schemes subsidise the phone itself; that is not established.

### Gaps
- **Udyam manufacturing split:** the number of *manufacturing* enterprises (vs services and trading) was not found in the 2026 releases. The MSME Annual Report's NSS 73rd-round table is the usual source and was not fetched. An idea page claims 8–10 million manufacturing MSMEs, but gives no source.
- **Unattributed global forecast:** an alternative global figure ($22.6 bn in 2025 rising to $61.0 bn by 2033 at 13.4%) appeared in a search summary, but I could not attribute it to a specific report. Don't use it.
- **No AI-visual-inspection market size specifically for India** (in value terms) was captured.

## 6. Which productivity metrics do factories use, and what numbers would a judge find credible for a phone-based system?

### Takeaway
Factories judge inspection through:
- **OEE**, where the quality term is 99.9% for world class;
- **PPM** at the customer (for example, AVTEC's bands);
- **first-pass yield**;
- detector-level metrics: **detection rate (recall), false-reject rate, image AUROC, latency or throughput, setup time**.

Training-free few-shot anomaly detection has reached **96.6% image AUROC on MVTec AD from a single good image** (AnomalyDINO, WACV 2025). On the harder MVTec AD 2 (2025), the best methods still score **below 60% AU-PRO**. So credible claims must be measured per category, under the team's own lighting, and reported at a fixed operating point.

### Cited Findings
- **OEE benchmarks [SECONDARY, practitioner references].** World-class OEE is 85% or more (Nakajima, TPM), made up of availability ≥90%, performance ≥95% and quality ≥99.9%. About 60% is typical for discrete manufacturers. — [OEE.com](https://www.oee.com/world-class-oee/); [Reliable Plant](https://www.reliableplant.com/Read/11785/overall-equipment-effectiveness)
- **Supplier PPM scoring [PRIMARY].** AVTEC scores 0 / 1–50 / 51–100 / 101–200 / 201–500 / >500 PPM as 25 / 20 / 15 / 10 / 5 / 0 points. — [AVTEC SQM](https://www.avtec.in/pdf/AVTEC%20Supplier%20Quality%20Manual.pdf)
- **Few-shot, training-free state of the art [PRIMARY, peer-reviewed].** AnomalyDINO uses DINOv2 patch features with nearest-neighbour matching, is "training-free", and raised one-shot MVTec AD image AUROC from 93.1% to **96.6%**. Published at WACV 2025 (oral). — [arXiv 2405.14529](https://arxiv.org/abs/2405.14529); [WACV paper](https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf)
- **MVTec AD [PRIMARY].** More than 5,000 high-resolution images in 15 object and texture categories. Training images are defect-free; test images include defects. Licence **CC BY-NC-SA 4.0 (no commercial use)**. — [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **MVTec AD 2 [PRIMARY, 2025].** More than 8,000 images in 8 scenarios. These include transparent or overlapping objects, dark-field and back-light illumination, high variance among normal parts, and very small defects. The best methods remain **below 60% average AU-PRO**. — [arXiv 2503.21622](https://arxiv.org/abs/2503.21622); [MVTec AD 2 page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
- **VisA [PRIMARY].** 10,821 images in 12 subsets, including 4 PCB subsets, licensed CC BY 4.0. — [spot-diff](https://github.com/amazon-science/spot-diff)
- **Industrial setup reference [VENDOR CLAIM].** Inspekto needs 20–30 good samples and 30–60 minutes. — [ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2019/04/10/inspekto-%E2%80%94-the-gold-mine-of-productivity/13232/); [RoboticsTomorrow](https://www.roboticstomorrow.com/news/2021/05/04/inspekto-launches-the-second-generation-of-autonomous-machine-vision-systems/16776/)
- **Device [PRIMARY, retail].** iQOO 15 with Snapdragon 8 Elite Gen 5: ₹72,999 (12 GB/256 GB) and ₹79,999 (16 GB/512 GB). — [91mobiles](https://www.91mobiles.com/hub/iqoo-15-launched-in-india-price-availability/)

### Inferences
Credible numbers for a phone-based system, and how each should be reported:
1. **Image AUROC per category** on MVTec AD or VisA, for 1, 4 and 16 good shots. This is directly comparable with AnomalyDINO's 96.6% at 1 shot. Claiming "≥ 95% AUROC on the matched category" is credible only if measured.
2. **Recall at a fixed false-reject rate**, for example recall at FRR = 1% or 5%. Factories care about the operating point, not AUROC. Compare with the human baseline of 85% hits and 35% false rejects (See 2015).
3. **Setup time** from "open app" to "armed", with the number of good samples used. Compare with 30–60 min and 20–30 samples for Inspekto.
4. **Latency** p50/p95 in ms per frame on the NPU, and **sustained FPS after 30–60 minutes**. Phones throttle thermally, and this is the phone's equivalent of human vigilance decrement, so measure it.
5. **Throughput** in parts/min at the station, limited by handling. Compare with a timed human doing the same parts.
6. **Repeatability:** decision flip rate when the same part is presented N times. Compare with humans reversing 23% of decisions.
7. **Cost per station** (₹73k vs ₹1.65–11 lakh) and **payback** in months of inspector wage.
8. **Offline proof:** run the full demo in airplane mode.

Honest caveats judges will respect:
- MVTec AD 2 shows that lighting changes, transparent parts and tiny defects break state-of-the-art methods. A fixed jig and diffuse light will likely be required.
- MVTec AD's licence is non-commercial. It is fine for a hackathon benchmark, but a product would need VisA (CC BY 4.0) or the team's own data.

### Gaps
- No published on-device latency of DINOv2-class anomaly detectors on the Snapdragon 8 Elite Gen 5 NPU was found in this research scope. It must be measured.
- No industry-standard acceptance thresholds (for example, a required recall at a given FRR) were found for AI inspection in Indian auto-component PPAP/IATF contexts.

## 7. X → Y claim sheet, and how each claim can be proven during the hackathon

### Takeaway
Every "Y" must be measured live or on a public benchmark. The "X" values below are sourced baselines. Build the pitch around four families of claims, each paired with a specific measurement: reliability (miss rate, false rejects, repeatability, fatigue), time (setup, latency, throughput), cost (per station, per inference, payback), and efficiency (PPM or escape reduction).

### Cited Findings (sourced "X" values; full citations in Sections 1–6)
**Reliability**

| Measure | Current state (X) | Source |
|---|---|---|
| Miss rate | 20–30% typical | [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf) |
| Expert hit rate / false rejects | 85% hits, 35% false rejects | [See 2015](https://www.sandia.gov/research/publications/details/visual-inspection-reliability-for-precision-manufactured-parts-2015-12-01/) |
| Fatigue | Detection down up to 40% in 30 min; 27% fewer hits between the first and second 15 minutes on automotive seals; 13–45% declines in field studies | [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf) |
| Repeatability | 23% of decisions reversed; per-inspector detection 43–100% | [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf) |
| Line speed | Doubling the pace raises misses from 23% to 30% | [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf) |

**Time**

| Measure | Current state (X) | Source |
|---|---|---|
| Setup | 30–60 min with 20–30 good samples (Inspekto); integration projects cost $5k–15k | [ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2019/04/10/inspekto-%E2%80%94-the-gold-mine-of-productivity/13232/); [Opsio](https://opsiocloud.com/knowledge-base/how-much-does-a-vision-inspection-system-cost/) |
| Human time per item | About 120 s for complex IC chips; vendor-claimed 38 s/part in automotive | [See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf); [iFactory](https://ifactoryapp.com/industries/automotive-manufacturing/ai-vs-manual-inspection-in-automotive-plants-speed-accuracy-and-cost) |

**Cost**

| Measure | Current state (X) | Source |
|---|---|---|
| Station hardware | ₹1.65 lakh (Keyence IV3), ₹1.8 lakh (Cognex SnAPP), ₹5.0 lakh (Keyence IV4), ≈ ₹11 lakh (Inspekto), ₹6.35–19.05 lakh (standard system) | [IndiaMART](https://www.indiamart.com/proddetail/keyence-vision-sensor-with-built-in-ai-iv3-series-2854380953255.html); [ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2019/04/10/inspekto-%E2%80%94-the-gold-mine-of-productivity/13232/); [Clearview](https://clearview-imaging.com/pages/how-much-does-machine-vision-actually-cost) |
| Cloud inference | 1 credit per image | [LandingLens](https://landinglens.docs.landing.ai/plans) |
| Inspector wage | ₹2.22–2.47 lakh/yr per shift | [Zimyo](https://www.zimyo.com/guides/minimum-wages/haryana/); [Indeed](https://in.indeed.com/career/quality-control-inspector/salaries) |
| Cost of quality | 15–20% of revenue (ASQ rule of thumb) | [IISE](https://www.iise.org/details.aspx?id=22118) |

**Efficiency / quality outcome**

| Measure | Current state (X) | Source |
|---|---|---|
| Customer PPM scoring | >500 PPM = 0 points; C=0 lot rejection; sorting at supplier cost | [AVTEC SQM](https://www.avtec.in/pdf/AVTEC%20Supplier%20Quality%20Manual.pdf) |
| Indian small-unit rejection | 15.5% before improvement | [ResearchGate](https://www.researchgate.net/publication/290120371_Reducing_rejectionrework_in_pressure_die_casting_process_by_application_of_dmaic_methodology_of_six_sigma) |
| OEE | 60% typical; world-class quality term 99.9% | [OEE.com](https://www.oee.com/world-class-oee/) |

### Inferences: proposed hackathon proof protocol (my design; each "Y" is to be measured, not assumed)

| # | Claim (X → Y) | How to produce the Y number during the hackathon | Output artefact |
|---|---|---|---|
| 1 | **Setup time:** 30–60 min with 20–30 good samples (Inspekto) → **Y min with N good frames** | Stopwatch from app launch to "armed" on a new part type. Three different operators, 3 runs each; report median and max. Film it. | Video plus timing table |
| 2 | **Miss rate:** 20–30% human (literature), and the *measured* local human miss rate → **app miss rate** | Build a physical test set: about 100 real auto parts (for example M8 bolts, nuts, washers, small castings) with about 10–20% seeded defects (scratch, dent, burr, missing thread, paint or rust spot). Blind human panel of 3–5 volunteers inspecting at a set pace vs the app. Randomise order and label ground truth beforehand. | Confusion matrices: human vs app |
| 3 | **False rejects:** 35% (See 2015 experts) → **app FRR at the chosen threshold** | Run on held-out *good* parts only (at least 50) and count flags. Fix the threshold on a separate calibration set first. | FRR with 95% confidence interval |
| 4 | **Benchmark accuracy:** training-free reference 96.6% image AUROC at 1 shot (AnomalyDINO, MVTec AD) → **app AUROC per category** | Run the on-device pipeline offline on MVTec AD (screw, metal nut or the closest categories) and VisA PCB1–4. Use k = 1, 4 and 16 good shots. Report image AUROC and recall at FRR = 1% and 5%. | Table vs published baseline |
| 5 | **Fatigue:** detection down 13–45% (up to 40% in 30 min) → **app accuracy at minute 0 vs minute 60** | Loop the same test set for 60 minutes of continuous operation. Log per-frame latency, FPS and decisions, and phone temperature. | Accuracy and FPS over time plot (shows thermal throttling honestly) |
| 6 | **Repeatability:** 23% of human decisions reversed → **app decision flip rate** | Present the same 20 parts 10 times each, with re-placement in the jig. Count decision changes. | Flip-rate % |
| 7 | **Throughput and latency:** human seconds per part (measured locally; vendor claim 38 s) → **app ms per frame (p50/p95) and parts/min** | App timestamps from frame capture to decision to beep. A second phone filming slow-motion at 240 fps can verify beep latency. Time a human inspecting the same 100 parts. | Latency histogram; parts/min |
| 8 | **Hardware cost:** ₹1.65–11 lakh per station → **₹72,999 phone (+ stand and light at measured cost)** | Show the itemised bill of materials with invoices or links. Express cost as months of an inspector's wage (≈ 3.9 months vs ≈ 59.5 for Inspekto). | Cost table |
| 9 | **Running cost:** cloud credits (1 per inference) → **₹0 per inference, offline** | Run the entire demo in airplane mode and show the inference counter. | Video |
| 10 | **Outcome:** escapes of 4,000–6,000 PPM (illustrative: 2% defect rate × 20–30% human miss) → **escapes = 2% × human miss × app miss** (app used as a second inspector, per Drury et al. 1986) | Compute from the measured miss rates in #2. Map the result onto AVTEC's PPM bands (>500 = 0 points). | One slide showing the PPM band change |

- **Framing advice from the evidence:**
  - Present Kaizen Eye as an **independent second inspector / poka-yoke aid** at the manual final-inspection station. The literature supports two-inspector schemes. In CSEP's three Indian case-study firms, automation led to worker redeployment, not retrenchment.
  - Keep the labour-replacement payback as a secondary number. The primary money story is escapes and PPM.
- **Honesty guard-rails:**
  - Label literature numbers as literature. Say that the human baseline you measure uses volunteers, not trained inspectors.
  - Report the operating threshold used.
  - Show at least one failure case (for example, a reflective part or changed lighting). MVTec AD 2 shows the whole field struggles there, so it builds credibility.

### Gaps
- No sourced data on how many images or minutes Cognex, Keyence or LandingLens deep-learning tools need for setup, so the only setup-time comparator is Inspekto's vendor claim.
- No sourced Indian MSME baseline for the human miss rate or time per part. The hackathon human panel is the proposed substitute and should be described as a proxy.
- Stand and lighting costs for the phone station were not sourced. The team should use its actual purchase prices.
