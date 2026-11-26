"""Offline advanced explanations for the pilot subject cards.

Basic answers stay concise. When a question explicitly asks for an advanced or
graduate-level treatment, this module expands the same card into a structured
explanation: definition, formal core, mechanism, framing and limits.

The content is deterministic and local. Wikipedia remains an internal evidence
check only; nothing in this module prints an engine or source badge.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime cycle
    from subject_rag import TopicCard


ADVANCED_REQUEST = re.compile(
    r"\b(?:advanced|graduate|in[- ]depth|detailed|comprehensive|university level)\b"
    r"|\bincluding (?:its |the )?formula\b"
    r"|\b(?:jordan|nilpotent)\b",
    re.I,
)


SUBJECT_LENSES: Dict[str, Dict[str, str]] = {
    "biology": {
        "formal": "No single universal equation; the useful core is a recognition -> response -> feedback loop.",
        "frame": "Treat the topic across molecular, cellular and organism scales rather than as an isolated fact.",
        "mechanism": "Identify the signal, the transduction path, the effector response and the feedback that terminates it.",
        "limits": "Cell type, developmental stage and environment change the outcome; apparent exceptions often reflect context or redundancy.",
    },
    "chemistry": {
        "formal": "The governing relation is the stated chemical law or balanced equation.",
        "frame": "Connect structure and bonding to energy, rate and observable properties.",
        "mechanism": "Track electrons, charge and energy through the steps; distinguish thermodynamic favourability from kinetic accessibility.",
        "limits": "Ideal, dilute or standard-state assumptions fail at high concentration, near phase changes or in complex media.",
    },
    "computer_science": {
        "formal": "Express the invariant or complexity relation that bounds the system.",
        "frame": "Analyse the abstraction, its interface and the resources it consumes.",
        "mechanism": "Follow the state transitions and data flow, then check correctness with an invariant or proof argument.",
        "limits": "Asymptotic bounds hide constants, memory layout and communication costs that dominate real systems.",
    },
    "economics": {
        "formal": "The core is the constraint or equilibrium relation linking incentives to outcomes.",
        "frame": "Identify agents, constraints, incentives and the adjustment process that leads to equilibrium or disequilibrium.",
        "mechanism": "Trace how a change in one price, policy or expectation propagates through choices and feedbacks.",
        "limits": "Results depend on assumptions about information, rationality and market structure; distributional effects matter.",
    },
    "geography": {
        "formal": "Describe the spatial process and the scale at which the relation holds.",
        "frame": "Analyse pattern and process together, across space and time.",
        "mechanism": "Trace flows of energy, water, sediment, people or capital through the system and identify feedbacks.",
        "limits": "Patterns are scale dependent; boundaries, data resolution and human decisions alter the conclusion.",
    },
    "history": {
        "formal": "There is no equation; the core is a causal claim supported by evidence.",
        "frame": "Separate context, event and interpretation; compare continuity with change.",
        "mechanism": "Trace how institutions, ideas, material conditions and individuals interacted to produce the outcome.",
        "limits": "Sources are partial and biased; counterfactuals and historiographical debate remain open.",
    },
    "math": {
        "formal": "The stated formula is the formal core; definitions supply the domain of validity.",
        "frame": "Move from definition to theorem to generalisation, and state the assumptions explicitly.",
        "mechanism": "Prove or derive the claim step by step, then test edge cases and limiting behaviour.",
        "limits": "Existence, convergence and domain restrictions can invalidate an otherwise plausible manipulation.",
    },
    "physics": {
        "formal": "State the governing law, the idealisation and the boundary conditions before solving.",
        "frame": "Connect conservation principles to the measured behaviour of the system.",
        "mechanism": "Apply the relevant force, field or energy relation and check units and limiting cases.",
        "limits": "Real systems depart from the ideal model through friction, dissipation, finite size or measurement resolution.",
    },
    "physics_branches": {
        "formal": "Specify the model, its governing equations and the regime in which they apply.",
        "frame": "Place the branch within physics by identifying its scales, objects and characteristic phenomena.",
        "mechanism": "Trace how the model's degrees of freedom, symmetries and interactions produce the observed behaviour.",
        "limits": "Approximations fail outside their domain; identify where the theory must be replaced or extended.",
    },
    "politics_law": {
        "formal": "The core is the rule, institution or competing claim being enforced.",
        "frame": "Analyse power, legitimacy and the rules that channel conflict.",
        "mechanism": "Trace how actors use institutions, incentives and procedures to secure or contest authority.",
        "limits": "Formal rules and lived practice often diverge; enforcement and access shape outcomes.",
    },
    "psychology": {
        "formal": "There is no single equation; the core is a testable mechanism linking stimulus, process and behaviour.",
        "frame": "Use multiple levels of analysis: biological, cognitive, social and behavioural.",
        "mechanism": "Identify the manipulation, the proposed psychological process and the measured outcome.",
        "limits": "Effects are probabilistic and context dependent; measurement, sampling and replication constrain claims.",
    },
}

DEFAULT_LENS: Dict[str, str] = {
    "formal": "State the governing relation or organising principle explicitly.",
    "frame": "Analyse definition, mechanism and boundary conditions together.",
    "mechanism": "Trace inputs through the system to the observed output.",
    "limits": "Context and assumptions can change the conclusion.",
}


# Topic-specific depth. Keys are (subject_key, normalised card title).
ADVANCED_NOTES: Dict[tuple, str] = {
    # Biology
    ("biology", "cell"): "Eukaryotic cells compartmentalise incompatible reactions into organelles. Membranes maintain ion gradients, the cytoskeleton gives shape and transport tracks, and signalling networks coordinate growth, division and death.",
    ("biology", "nucleus"): "The nucleus separates transcription from translation, allowing RNA processing before export. Nuclear pores regulate traffic, chromatin organisation controls access to DNA, and repair pathways protect the genome during replication and damage.",
    ("biology", "photosynthesis"): "Light reactions in thylakoid membranes convert photons into ATP and NADPH through an electron-transport chain and chemiosmosis. The Calvin cycle then fixes carbon dioxide with Rubisco; photorespiration and light intensity limit net gain.",
    ("biology", "dna"): "Complementary base pairing lets each strand template the other, which is the basis of semi-conservative replication. Antiparallel strand geometry, helicase, polymerase proofreading and chromatin packaging determine accuracy and accessibility.",
    ("biology", "protein synthesis"): "Transcription produces mRNA, and the ribosome reads codons while tRNA molecules deliver matching amino acids. Initiation, elongation and termination factors control fidelity; folding and post-translational modification determine the functional product.",
    ("biology", "chromosome"): "Chromosomes package long DNA molecules with histones into condensable units. Centromeres attach to the spindle, telomeres protect ends, and recombination during meiosis reshuffles alleles while segregation preserves chromosome number.",
    ("biology", "meiosis"): "One round of replication is followed by two divisions that separate homologues and then sister chromatids. Crossing over in prophase I and independent assortment generate variation; cohesion failure causes nondisjunction.",
    ("biology", "evolution"): "Allele frequencies change through selection, drift, mutation and gene flow. Selection acts on phenotypes in a particular environment, while evolution is measured in populations across generations, so fitness is context dependent.",
    ("biology", "mendelian inheritance"): "Alleles segregate during gamete formation, and loci on different chromosomes assort independently. Linkage, incomplete dominance, penetrance and epistasis modify simple ratios without overturning the underlying segregation mechanism.",
    ("biology", "ecosystem"): "Energy enters through producers and dissipates at each trophic transfer, while nutrients cycle between biotic and abiotic pools. Stability depends on diversity, functional redundancy and the disturbance regime.",
    ("biology", "biodiversity"): "Biodiversity includes genetic, species and functional dimensions. Richness and evenness capture different aspects; effects on stability depend on redundancy, trophic structure and the scale of observation.",
    ("biology", "bacteria"): "Small genomes and rapid reproduction allow fast adaptation. Cell wall structure, membranes and efflux pumps define niches, while conjugation, transformation and transduction spread traits such as resistance.",
    ("biology", "fungi"): "Hyphal growth and externally secreted enzymes let fungi exploit complex substrates. Chitin walls and ergosterol membranes are selective drug targets, and spores plus dormancy support dispersal and survival.",
    ("biology", "animal tissue"): "Tissues integrate specialised cells through junctions and extracellular matrix. Epithelial polarity, connective support, muscle contraction and neural signalling are coordinated by stem-cell turnover and local cues.",
    ("biology", "circulatory system"): "Pressure gradients drive flow, with cardiac output equal to heart rate times stroke volume. Vascular resistance and capillary exchange distribute nutrients, while baroreflexes stabilise perfusion under perturbation.",
    ("biology", "nervous system"): "Neurons encode information in action-potential timing and synaptic strength. Ion gradients, myelination and transmitter systems shape speed and reliability; synaptic plasticity provides a cellular basis for learning.",
    ("biology", "immune system"): "Antigen-presenting cells process proteins and display peptides on MHC molecules. Pattern-recognition receptors detect pathogens, complement and phagocytes amplify the response, and adaptive immunity selects rare B and T clones that leave memory cells for faster re-exposure.",
    ("biology", "hormone"): "Endocrine signals travel in blood but act only on cells with the matching receptor. Ligand binding, second messengers and feedback axes tune amplitude and duration; pulsatile release can change the response to the same concentration.",
    ("biology", "heart"): "The cardiac cycle coordinates electrical conduction, valve timing and pressure change. The Frank-Starling relation links preload to output, and coronary supply must match the oxygen demand of the contracting muscle.",
    ("biology", "reproduction"): "Gametogenesis, mating and embryonic development are coordinated by hormonal axes. Success depends on timing, resource allocation, genetic compatibility and the survival of offspring to reproductive age.",
    ("biology", "ecology"): "Population and community patterns emerge from births, deaths, dispersal and species interactions. Density dependence, niche partitioning and disturbance scale local events up to regional structure.",
    ("biology", "adaptation"): "Adaptations are traits shaped by past selection for a function. Distinguish current utility from historical origin; trade-offs, pleiotropy and physical constraints prevent perfection.",
    ("biology", "fossil"): "Fossils form when hard parts or traces survive burial and mineralisation. Stratigraphy and radiometric dating constrain age, but the record is biased toward certain environments and organisms.",
    ("biology", "biotechnology"): "Engineered biological systems combine modular parts, chassis organisms and selection schemes. Expression burden, genetic stability, containment and off-target effects constrain reliable design.",
    ("biology", "microbiology"): "Microbial communities are structured by resource gradients, syntrophy and quorum sensing. Culturing bias means molecular methods reveal much of the diversity that laboratory isolation misses.",
    # Chemistry
    ("chemistry", "atom"): "Electrons occupy quantised orbitals rather than fixed paths. Shielding and penetration set effective nuclear charge, which explains periodic trends in size, ionisation energy and reactivity.",
    ("chemistry", "element"): "Element identity is proton number. Isotopes share chemistry but differ in mass and nuclear stability; cosmic abundance reflects nucleosynthesis and geochemical behaviour.",
    ("chemistry", "ion"): "Ion formation balances ionisation energy and electron affinity against lattice or solvation stabilisation. Charge density controls hydration strength, mobility and interaction with counterions.",
    ("chemistry", "periodic table"): "Periodicity comes from repeating valence configurations. Effective nuclear charge and shielding drive trends, while filled and half-filled subshells explain important exceptions.",
    ("chemistry", "ionic bond"): "Ionic bonding balances electrostatic lattice enthalpy against the cost of forming ions. Structure, ionic radius and charge determine lattice energy and therefore melting point and solubility.",
    ("chemistry", "metallic bond"): "Delocalised electrons occupy bands and bind the cation lattice. Band filling and electron density explain conductivity, malleability and the range of melting points.",
    ("chemistry", "hybridization"): "Hybrid orbitals are a localised description of orbital mixing that matches observed geometry. sp, sp2 and sp3 character correlate with bond angles; delocalised systems require a molecular-orbital view.",
    ("chemistry", "molar mass"): "Molar mass connects microscopic formula units to measurable mass through Avogadro's constant. Isotopic distributions make ordinary atomic weights averages rather than single masses.",
    ("chemistry", "stoichiometry"): "A balanced equation encodes conservation of atoms and determines the limiting reagent. Side reactions, incomplete conversion and purification losses reduce actual yield below theoretical yield.",
    ("chemistry", "concentration"): "Concentration is amount per volume, but activity governs real-solution thermodynamics. Dilution, mixing and temperature require consistent units and attention to volume changes.",
    ("chemistry", "acid"): "Brønsted-Lowry acidity is proton donation; strength depends on conjugate-base stability and the solvent. Lewis acidity broadens the idea to electron-pair acceptance.",
    ("chemistry", "ph"): "pH is the negative logarithm of hydrogen-ion activity, not exactly concentration. Buffers resist change near their pKa, and temperature and ionic strength shift measured values.",
    ("chemistry", "oxidation"): "Oxidation is electron loss or an increase in oxidation state. Half-reactions conserve mass and charge; the driving force depends on electrode potentials and reaction conditions.",
    ("chemistry", "redox"): "Electron transfer is tracked with oxidation numbers and balanced half-reactions. Cell potential predicts spontaneity through Gibbs energy, but kinetics can block a thermodynamically favoured reaction.",
    ("chemistry", "gas laws"): "Ideal behaviour assumes negligible intermolecular forces and particle volume. Real gases deviate near condensation or high pressure; the compressibility factor measures that departure.",
    ("chemistry", "boyle s law"): "At constant temperature, pressure varies inversely with volume for a fixed amount of ideal gas. It is a limiting law; attractions and finite molecular size matter at high density.",
    ("chemistry", "thermochemistry"): "Enthalpy tracks heat at constant pressure and is a state function. Hess's law and bond enthalpies estimate reaction heat from pathways and average bond energies.",
    ("chemistry", "chemical entropy"): "Entropy reflects the number of accessible microstates and the dispersal of energy. The entropy change of the surroundings links reaction enthalpy to spontaneity through Gibbs free energy.",
    ("chemistry", "chemical equilibrium"): "Equilibrium is dynamic equality of forward and reverse rates. Q predicts direction relative to K, and a catalyst changes the approach rate without changing K.",
    ("chemistry", "rate of reaction"): "Rate laws must be determined experimentally except for elementary steps. Activation energy, collision geometry and the pre-exponential factor control the Arrhenius behaviour.",
    ("chemistry", "solution"): "Solvation competes with solute-solute and solvent-solvent attractions. This balance controls solubility, colligative properties and deviations from ideal behaviour.",
    ("chemistry", "colloid"): "Colloids are kinetically stable dispersions. Surface charge, steric layers and DLVO balance determine aggregation, while Brownian motion opposes settling.",
    ("chemistry", "hydrocarbon"): "Saturation, branching and ring strain control physical properties and reaction pathways. Alkanes favour substitution; alkenes and aromatics favour addition or electrophilic substitution respectively.",
    ("chemistry", "aldehyde and ketone"): "The carbonyl group polarises the C=O bond, making carbon electrophilic. Nucleophiles attack that centre; aldehydes are usually more reactive because they are less sterically hindered and less electron-rich.",
    ("chemistry", "polymer"): "Chain length distribution, architecture and intermolecular forces determine properties. Polymerisation mechanism controls stereochemistry, branching and dispersity.",
    # Computer Science
    ("computer_science", "algorithm"): "An algorithm is a finite, precise procedure. Correctness proofs use loop invariants; complexity analysis measures how resource needs grow with input size, independent of a particular machine.",
    ("computer_science", "array"): "Contiguous storage gives constant-time indexing but expensive insertion and deletion. Cache locality often dominates real performance more than the asymptotic operation count.",
    ("computer_science", "stack"): "LIFO access restricts operations to the top, which simplifies call stacks, expression evaluation and depth-first search. Amortised analysis explains the cost of dynamic resizing.",
    ("computer_science", "tree"): "Hierarchical branching enables logarithmic search when the tree is balanced. Traversal order encodes structure, and imbalance degrades worst-case performance.",
    ("computer_science", "graph"): "Vertices and edges model relationships. Adjacency lists and matrices trade memory for query speed; shortest paths and connectivity depend on direction and edge weights.",
    ("computer_science", "sorting"): "Comparison sorts need at least n log n comparisons in the worst case. Stability, in-place behaviour and adaptivity matter as much as average speed in practice.",
    ("computer_science", "binary search"): "Each step halves the candidate interval, giving logarithmic time. Correctness relies on sorted order and careful invariant maintenance at the boundaries.",
    ("computer_science", "complexity"): "Big-O describes growth up to constant factors. Worst, average and amortised cases differ, and memory or communication can be the true bottleneck.",
    ("computer_science", "object oriented programming"): "Objects bundle state and behaviour behind interfaces. Encapsulation protects invariants, while polymorphism enables substitution at the cost of indirection.",
    ("computer_science", "inheritance"): "Subclasses reuse and specialise behaviour. The Liskov substitution principle constrains safe overrides, and deep hierarchies increase coupling.",
    ("computer_science", "encapsulation"): "Hidden representation protects invariants and reduces coupling. Wide getters that expose internals can quietly undo the benefit.",
    ("computer_science", "sql"): "SQL is declarative: it describes the desired result, not the procedure. The optimizer chooses plans using indexes and statistics; NULL and join semantics require care.",
    ("computer_science", "indexing"): "Indexes trade storage and write cost for faster reads. Selectivity, cardinality and covering determine usefulness, and composite column order matters.",
    ("computer_science", "process"): "A process owns an address space and resources. The scheduler switches contexts; isolation and interprocess communication shape reliability and performance.",
    ("computer_science", "memory management"): "Allocation, reuse and reclamation determine locality and lifetime. Garbage collectors trade pause time, throughput and footprint, while leaks are unreachable lifetime errors.",
    ("computer_science", "computer network"): "Layered protocols hide heterogeneity. Routing, congestion control and error recovery interact, and throughput is bounded by bottlenecks and round-trip time.",
    ("computer_science", "http"): "Request-response semantics sit over TCP or QUIC. Methods, status codes and headers define caching and idempotence; statelessness pushes state to clients or stores.",
    ("computer_science", "encryption"): "Confidentiality relies on keys and hard problems. Authenticated encryption also provides integrity; nonce reuse and poor key management cause most real failures.",
    ("computer_science", "machine learning"): "Models learn mappings from data by optimising a loss. Generalisation depends on bias, variance and data distribution; regularisation controls capacity.",
    ("computer_science", "overfitting"): "Low training error with poor test error signals memorisation. Capacity, noise and sample size interact; validation and regularisation mitigate but do not remove the risk.",
    ("computer_science", "compiler"): "Compilers pass through lexing, parsing, semantic analysis and optimisation. Intermediate representations decouple front ends from target architectures.",
    ("computer_science", "programming language"): "Syntax, semantics and type systems define abstraction and safety. Evaluation order and memory model change behaviour even when code looks similar.",
    ("computer_science", "python"): "Dynamic typing and runtime introspection speed development. The GIL, object overhead and interpreted loops shape performance; idiom matters more than micro-optimisation.",
    ("computer_science", "web browser"): "Browsers parse HTML and CSS into trees, execute JavaScript and composite frames. The same-origin policy and rendering pipeline determine security and performance.",
    ("computer_science", "cloud computing"): "Elastic resources are provisioned through abstractions. Availability zones, scaling policies and cost models matter; distributed failure is normal rather than exceptional.",
    # Economics
    ("economics", "supply and demand"): "Equilibrium emerges where buyer and seller plans reconcile. Shifts of a curve differ from movements along it, and elasticity determines incidence and adjustment speed.",
    ("economics", "opportunity cost"): "The true cost is the best forgone alternative. Accounting costs miss implicit value, and sunk costs should not affect a forward-looking choice.",
    ("economics", "gdp"): "GDP measures production, not wellbeing. Nominal versus real, expenditure versus income and informal activity all affect interpretation.",
    ("economics", "unemployment"): "Unemployment contains frictional, structural and cyclical components. Natural-rate and hysteresis arguments explain why long spells persist after a shock.",
    ("economics", "monetary policy"): "Central banks influence demand through rates and expectations. Transmission lags and the effective lower bound complicate stabilisation.",
    ("economics", "compound interest"): "Growth compounds on prior growth, so time is exponentially powerful. Effective annual rates depend on compounding frequency; real returns must subtract inflation and tax.",
    ("economics", "consumer surplus"): "Surplus is the area between willingness to pay and the price paid. Price discrimination transfers surplus rather than necessarily destroying total surplus.",
    ("economics", "marginal utility"): "Rational choice equalises marginal utility per unit cost. Diminishing utility and the budget constraint generate demand, while behavioural biases create systematic deviations.",
    ("economics", "perfect competition"): "Price-taking, homogeneous goods and free entry drive long-run economic profit toward zero. The information and scale assumptions are strong.",
    ("economics", "oligopoly"): "A few firms make strategic decisions. Game theory, concentration and entry barriers determine outcomes; collusion can be profitable but unstable.",
    ("economics", "price discrimination"): "Charging different prices requires market power and market segmentation. It can raise output while transferring surplus; arbitrage limits implementation.",
    ("economics", "externality"): "Private costs diverge from social costs. Taxes or tradable permits internalise the gap, but measurement and distribution remain difficult.",
    ("economics", "tax"): "Taxes change relative prices and behaviour. Incidence depends on elasticity rather than statutory burden, and deadweight loss grows with the distortion.",
    ("economics", "budget deficit"): "A deficit is a flow; debt is the accumulated stock. Sustainability depends on the interest-growth differential and the primary balance.",
    ("economics", "exchange rate"): "Rates reflect relative prices, returns and expectations. Floating regimes absorb shocks; fixed regimes require credibility and reserves.",
    ("economics", "comparative advantage"): "Trade follows relative opportunity costs, not absolute productivity. Specialisation raises joint output even when one partner is absolutely more productive.",
    ("economics", "trade"): "Trade expands consumption possibilities through specialisation. Transport costs, institutions and adjustment frictions shape who gains and who loses.",
    ("economics", "quota"): "A quota restricts quantity and creates scarcity rents. Unlike a tariff, the licence allocation decides who receives the rent.",
    ("economics", "central bank"): "Central banks set policy rates and support stability. Independence, credibility and communication shape expectations; lender-of-last-resort action creates moral hazard.",
    ("economics", "quantity theory of money"): "MV = PY links money, velocity, price level and output. Stable velocity is an empirical assumption, not a law.",
    ("economics", "business cycle"): "Fluctuations arise from demand shocks, supply shocks and financial frictions. Multipliers and accelerators propagate disturbances across sectors.",
    ("economics", "depression"): "A depression is a deep, prolonged downturn. Weak demand, debt deflation and policy constraints prolong it; recovery often requires balance-sheet repair.",
    ("economics", "development economics"): "Development involves income, health, education and institutions. Diagnostics must separate proximate input gaps from deeper political or informational constraints.",
    ("economics", "income inequality"): "Inequality reflects factor returns, education, institutions and policy. Pretax versus posttax and income versus wealth give different pictures.",
    ("economics", "labor market"): "Wages clear markets only under strong assumptions. Search frictions, bargaining, efficiency wages and institutions determine employment outcomes.",
    # Geography
    ("geography", "continent"): "Continents are large landmasses defined by geology and convention. Plate motion and sea-level change reshape their boundaries over deep time.",
    ("geography", "mountain"): "Mountains form through tectonics, volcanism or erosion. Uplift and denudation compete, and elevation reorganises climate, soils and settlement.",
    ("geography", "desert"): "Aridity follows from subtropical highs, rain shadows or continentality. Deserts are not lifeless; episodic water and adaptation structure their ecosystems.",
    ("geography", "climate"): "Climate is the statistics of weather over time. Ice, cloud and carbon feedbacks amplify or damp forcing, while circulation redistributes energy.",
    ("geography", "atmosphere"): "The atmosphere is a stratified fluid. Convection, radiation and planetary rotation drive circulation; composition determines greenhouse forcing.",
    ("geography", "lithosphere"): "The lithosphere is the rigid outer shell. Isostatic balance and plate motion explain topography, earthquakes and long-term crustal recycling.",
    ("geography", "plate tectonics"): "Plates move over the weaker asthenosphere. Convergence, divergence and transform motion produce hazards and recycle crust over geological time.",
    ("geography", "volcano"): "Magma rises through buoyancy, volatile exsolution and tectonic setting. Viscosity and gas content control whether an eruption is effusive or explosive.",
    ("geography", "hurricane"): "Warm oceans and low wind shear fuel tropical cyclones. Latent heat drives the engine; steering flows and shear determine track and intensity.",
    ("geography", "latitude"): "Latitude controls solar angle and day length. Altitude, currents and continentality modify the broad climate zones it produces.",
    ("geography", "equator"): "The equator receives high year-round insolation, and the migrating ITCZ drives seasonality. The Coriolis effect vanishes there and grows poleward.",
    ("geography", "time zone"): "Time zones standardise solar time across longitude. Political and economic choices often bend the ideal fifteen-degree bands.",
    ("geography", "scale"): "Patterns change with scale. Processes dominant locally may be invisible globally, and the modifiable areal unit problem affects statistical inference.",
    ("geography", "river delta"): "Deltas form where deposition outpaces sediment removal. Subsidence, sea-level rise and upstream dams threaten their stability.",
    ("geography", "groundwater"): "Aquifers store water in pore spaces. Recharge, permeability and contamination determine sustainable yield; cones of depression signal overuse.",
    ("geography", "erosion"): "Erosion detaches and transports material. Water, wind and ice dominate in different climates; slope, vegetation and substrate set the rate.",
    ("geography", "soil"): "Soil forms from parent material through weathering, biology and time. Horizons record gains and losses, and degradation can outpace formation.",
    ("geography", "renewable resource"): "Renewability depends on replenishment relative to use. Solar and wind are flows; forests and fisheries can be depleted despite being renewable.",
    ("geography", "population density"): "Density reflects births, deaths and migration divided by area. Uneven distribution follows opportunity, hazards and policy.",
    ("geography", "migration"): "Migration responds to wage gaps, conflict, networks and policy. Selection and integration shape impacts at both origin and destination.",
    ("geography", "capital city"): "Capital location reflects political, historical and geographic forces. Planned capitals differ from organic ones in legitimacy and function.",
    ("geography", "sovereignty"): "Sovereignty is recognised authority over territory. Globalisation, federal arrangements and non-state actors contest its practice.",
    ("geography", "biome"): "Biomes track climate, especially temperature and moisture. Fire, soils and herbivory can shift the boundary between alternative stable states.",
    ("geography", "savanna"): "Savannas persist through a fire-grass-herbivore feedback. Tree cover depends on rainfall, fire frequency and browsing pressure.",
    ("geography", "coral reef"): "Reefs build calcium carbonate structures within narrow thermal and light limits. Bleaching, acidification and pollution threaten net accretion.",
    # History
    ("history", "ancient egypt"): "Nile floods supported surplus agriculture and centralised administration. Writing, monumental building and afterlife belief reinforced state legitimacy.",
    ("history", "indus valley"): "Urban planning and long-distance trade imply sophisticated civic organisation. The undeciphered script and uncertain political structure limit interpretation.",
    ("history", "ancient greece"): "City-states produced competing political experiments. Philosophy, theatre and science flourished alongside slavery and exclusion.",
    ("history", "maurya empire"): "Centralised administration and Ashokan edicts projected authority. Legitimacy combined conquest with Buddhist patronage and welfare measures.",
    ("history", "mughal empire"): "Mughal rule joined Persianate administration with local elites. Revenue systems, marriage alliances and architecture integrated diverse regions.",
    ("history", "islamic golden age"): "Translation movements preserved and extended Greek, Persian and Indian learning. Scholars advanced algebra, optics and medicine through connected networks.",
    ("history", "industrial revolution"): "Mechanisation, coal and factory organisation transformed production. Urbanisation and new labour relations created both wealth and conflict.",
    ("history", "american revolution"): "Colonial resistance to imperial policy became a struggle over representation and sovereignty. Ideas and material interests reinforced one another.",
    ("history", "world war i"): "Alliance systems, imperial rivalry and mobilisation plans turned a regional crisis into global war. Industrial technology changed the scale of destruction.",
    ("history", "cold war"): "Rivalry combined ideological, military and economic competition. Deterrence, proxy wars and arms control shaped the international order.",
    ("history", "partition of india"): "Partition followed decolonisation and communal politics. Mass displacement and violence shaped South Asian states for decades.",
    ("history", "alexander the great"): "Rapid conquest created a vast but fragile empire. Cultural mixing followed military expansion, while succession disputes fragmented the territory.",
    ("history", "cleopatra"): "Ptolemaic rule combined Egyptian and Hellenistic traditions. Cleopatra's Roman alliances were strategic responses to dynastic and imperial pressure.",
    ("history", "nelson mandela"): "Mandela became a symbol of resistance to apartheid and of negotiated transition. Institution-building and reconciliation mattered alongside his imprisonment.",
    ("history", "silk road"): "Trade networks moved goods, ideas and disease across Eurasia. Intermediaries, caravanserai and political protection made long-distance exchange possible.",
    ("history", "capitalism history"): "Capitalism developed through merchant trade, industrialisation and finance. Property rights, markets and state power co-evolved rather than appearing at once.",
    ("history", "colonialism"): "Colonial rule restructured economies, borders and knowledge systems. Extraction and resistance shaped both colony and metropole.",
    ("history", "nation state"): "The nation-state links sovereignty to an imagined community. Borders, language and collective memory are politically constructed.",
    ("history", "human rights history"): "Human rights emerged from Enlightenment thought, abolition and post-war struggles. Declarations state ideals; implementation depends on institutions.",
    ("history", "civil rights movement"): "Grassroots organising, litigation and protest forced legal change. Federal action and local resistance interacted throughout the campaign.",
    ("history", "printing press"): "Movable type lowered the cost of texts and standardised knowledge. It accelerated religious reform and scientific controversy.",
    ("history", "bronze age"): "Bronze tools and weapons improved durability and craft specialisation. Trade in copper and tin connected distant societies.",
    ("history", "middle ages"): "Feudal, ecclesiastical and urban institutions coexisted. The period combined local fragmentation with long-distance exchange and intellectual continuity.",
    ("history", "enlightenment"): "Enlightenment thought stressed reason, scepticism and natural rights. Salons, journals and print networks spread ideas unevenly across Europe and beyond.",
    ("history", "united nations"): "The UN supplies a forum and normative framework. Its effectiveness depends on great-power cooperation, funding and implementation.",
    # Mathematics
    ("math", "derivative"): "The derivative is the best linear approximation at a point. It measures sensitivity, gives the slope of a tangent and drives optimisation through gradients.",
    ("math", "limit"): "A limit describes approach rather than evaluation at the point. Epsilon-delta definitions remove ambiguity about what approaching means.",
    ("math", "matrix"): "A matrix represents a linear map after bases are chosen. Rank, determinant and eigenvalues summarise different aspects of the same map.",
    ("math", "eigenvalue"): "Eigenvectors keep their direction under a transformation. Eigenvalues govern powers, stability and whether the matrix can be diagonalised.",
    ("math", "nilpotent matrix"): "A nilpotent matrix has all eigenvalues equal to zero and some power equal to the zero matrix. Its Jordan form consists of strictly upper triangular blocks.",
    ("math", "schur complement"): "Block Gaussian elimination produces the Schur complement. It connects determinants, matrix inverses and conditional covariance in Gaussian models.",
    ("math", "trace"): "Trace is the sum of diagonal entries and, over complex numbers, the sum of eigenvalues. It is invariant under change of basis and cyclic in products.",
    ("math", "linear transformation"): "Linear maps preserve addition and scalar multiplication. Kernel and image capture what is lost and what is reachable; bases change representation, not the map.",
    ("math", "cross product"): "In three dimensions the cross product is perpendicular to both inputs and has magnitude equal to their parallelogram area. The wedge product generalises it to other dimensions.",
    ("math", "probability"): "Probability formalises uncertainty with measure theory. Independence and conditional probability are distinct, and Bayes' rule updates beliefs with evidence.",
    ("math", "expected value"): "Expectation is a probability-weighted average. Linearity holds without independence, but existence requires the relevant integral or sum to converge.",
    ("math", "standard deviation"): "Standard deviation is the square root of variance and shares the data's units. It is not the mean absolute deviation and is sensitive to outliers.",
    ("math", "binomial theorem"): "The theorem expands powers of sums with combinatorial coefficients. Those coefficients count choices and link algebra to binomial probabilities.",
    ("math", "combination"): "Combinations count unordered selections. Order matters for permutations, and symmetry gives n choose k equal to n choose n minus k.",
    ("math", "gcd and lcm"): "Euclid's algorithm computes the gcd efficiently via repeated remainders. For positive integers, gcd times lcm equals the product; prime factorisation makes this clear.",
    ("math", "complex number"): "Complex numbers add an imaginary unit with i squared equal to minus one. Modulus and argument give polar form, and Euler's formula connects exponentials with trigonometry.",
    ("math", "quadratic equation"): "The discriminant determines whether roots are real and distinct. Completing the square explains the formula and links coefficients to the vertex and symmetry axis.",
    ("math", "logarithm"): "Logarithms invert exponentiation. They convert products to sums and powers to multiples; changing base only introduces a constant factor.",
    ("math", "sequence"): "A sequence is an ordered function on the natural numbers. Convergence, monotonicity and boundedness are related but distinct properties.",
    ("math", "taylor series"): "Taylor polynomials approximate smooth functions locally using derivatives at a point. The remainder controls error, and convergence is not guaranteed everywhere.",
    ("math", "partial derivative"): "A partial derivative varies one input while holding the others fixed. The total derivative accounts for indirect dependence among variables.",
    ("math", "optimization"): "Optimisation seeks extrema, often under constraints. First-order conditions are necessary; convexity or second-order tests are needed for sufficiency.",
    ("math", "logic"): "Logic studies valid inference. Syntax, semantics and proof systems are distinct, with soundness and completeness linking proof to truth.",
    ("math", "number system"): "Number systems extend closure properties. Integers handle subtraction, rationals handle division, reals handle limits, and complexes handle all polynomial roots.",
    ("math", "percentage"): "A percentage is a ratio per hundred. Successive percentage changes multiply rather than add, and the chosen base changes interpretation.",
    # Physics
    ("physics", "newton s first law"): "Inertia means velocity persists without a net force. The law identifies inertial reference frames rather than describing a new interaction.",
    ("physics", "newton s third law"): "Forces occur in equal and opposite pairs on different bodies, so they do not cancel on a single body. The paired impulses conserve total momentum.",
    ("physics", "mass and weight"): "Mass measures inertia; weight is the gravitational force on that mass. Weight changes with local gravitational field while mass does not.",
    ("physics", "work"): "Work transfers energy through force acting along displacement. Only the parallel component contributes, and path dependence matters for non-conservative forces.",
    ("physics", "power"): "Power is the rate of energy transfer. Average and instantaneous power differ, and efficiency compares useful output with total input.",
    ("physics", "impulse"): "Impulse is force integrated over time and equals the change in momentum. Peak force depends strongly on stopping time, not only on momentum change.",
    ("physics", "centripetal force"): "Circular motion requires inward acceleration of v squared over r. Centripetal force is a role played by tension, gravity or friction, not a separate interaction.",
    ("physics", "escape velocity"): "Escape speed follows from equating kinetic and gravitational potential energy. The simple result ignores atmosphere, rotation and powered ascent.",
    ("physics", "projectile motion"): "Horizontal and vertical motions decouple under constant gravity. Air resistance breaks the symmetry and shifts the maximum-range angle below 45 degrees.",
    ("physics", "oscillation"): "Oscillations arise when a restoring force opposes displacement. Damping removes energy, while periodic driving can produce resonance.",
    ("physics", "wave"): "Waves transfer energy without bulk transport of matter. Wavelength, frequency and speed are linked, and superposition produces interference.",
    ("physics", "reflection"): "Specular reflection obeys equal angles relative to the normal. Phase change on reflection depends on the refractive-index boundary.",
    ("physics", "lens"): "A lens focuses light by refraction. Focal length, aperture and aberration determine image quality; the thin-lens equation is a paraxial approximation.",
    ("physics", "snell s law"): "Refraction follows from phase continuity and a change in wave speed. Total internal reflection occurs beyond the critical angle when moving to a lower-index medium.",
    ("physics", "coulomb s law"): "Electrostatic force follows inverse-square scaling and superposes across charges. The medium's permittivity and geometry determine the net effect.",
    ("physics", "electric potential"): "Potential is potential energy per unit charge. Only differences are physical, and the electric field is the negative gradient of potential.",
    ("physics", "resistance"): "Resistance depends on geometry and material resistivity. Temperature and microscopic scattering set its value; Ohm's law is a linear empirical regime.",
    ("physics", "series and parallel circuits"): "Series elements share current; parallel elements share voltage. Equivalent values follow from Kirchhoff's current and voltage laws.",
    ("physics", "faraday s law"): "A changing magnetic flux induces an electromotive force. Lenz's law fixes the sign, and generators and transformers rely on this coupling.",
    ("physics", "ac and dc"): "DC keeps a fixed polarity; AC oscillates. RMS values make AC power comparable to DC, and reactive components introduce phase shifts.",
    ("physics", "heat"): "Heat is energy in transit because of a temperature difference. It is not stored; internal energy is the state variable.",
    ("physics", "entropy"): "Entropy measures energy dispersal and multiplicity. The second law is statistical, and reversible processes leave total entropy unchanged.",
    ("physics", "kinetic theory"): "Pressure arises from molecular collisions with boundaries. Temperature measures average translational kinetic energy, and dense gases deviate from ideal behaviour.",
    ("physics", "quantum mechanics"): "States are vectors in Hilbert space and observables are operators. Measurement outcomes are probabilistic, and context matters.",
    ("physics", "big bang theory"): "The model describes expansion from a hot, dense early state, not an explosion into pre-existing space. The cosmic microwave background and light-element abundances are key evidence.",
    # Physics branches
    ("physics_branches", "quantum field theory"): "Fields are the fundamental objects and particles are their quantised excitations. Symmetry principles fix interaction terms, renormalisation handles scale dependence, and perturbation theory works when the coupling is small.",
    ("physics_branches", "general relativity"): "Gravity is spacetime curvature sourced by energy and momentum. Geodesics replace gravitational force, and solutions such as Schwarzschild and Kerr describe strong-field regimes where Newtonian gravity fails.",
    ("physics_branches", "condensed matter physics"): "Many-body interactions produce phases and excitations not visible in isolated particles. Symmetry breaking, band structure and collective modes explain magnets, superconductors and semiconductors.",
    ("physics_branches", "particle physics"): "The Standard Model organises quarks, leptons and gauge bosons by internal symmetries. Accelerators probe short distances, and cross-sections connect theoretical amplitudes to detector counts.",
    ("physics_branches", "nuclear physics"): "Nuclei are bound quantum many-body systems. Binding energy, shell structure, decay channels and reaction thresholds determine stability and the energy released by fission or fusion.",
    ("physics_branches", "astrophysics"): "Stars and galaxies are natural laboratories governed by gravity, radiation and plasma physics. Spectra, luminosity and orbital motion constrain mass, composition and distance.",
    ("physics_branches", "cosmology"): "The Friedmann equations link expansion history to energy content. Lambda-CDM fits the cosmic microwave background, large-scale structure and supernova data, while dark matter and dark energy remain open problems.",
    ("physics_branches", "plasma physics"): "Charged particles collectively respond to electromagnetic fields. Debye shielding, plasma oscillations and magnetic confinement control behaviour in fusion devices and astrophysical plasmas.",
    ("physics_branches", "fluid dynamics"): "Conservation of mass, momentum and energy yields the Navier-Stokes equations. Reynolds number distinguishes laminar from turbulent flow, and boundary conditions determine the solution regime.",
    ("physics_branches", "optics"): "Geometrical optics follows rays when wavelengths are small; wave optics handles interference and diffraction. Maxwell's equations unite reflection, refraction and polarisation, while photon statistics require quantum optics.",
    ("physics_branches", "atomic physics"): "Electrons occupy quantised orbitals set by the Coulomb potential and boundary conditions. Fine structure, hyperfine splitting and selection rules arise from small interactions and symmetry constraints.",
    ("physics_branches", "molecular physics"): "Molecules have coupled electronic, vibrational and rotational energy scales. Spectroscopy resolves these levels, and the Born-Oppenheimer approximation separates fast electronic motion from slower nuclear motion.",
    ("physics_branches", "statistical mechanics"): "Macroscopic thermodynamics emerges from counting microstates. Ensembles and partition functions connect probabilities to free energy, and fluctuations become important in small systems.",
    ("physics_branches", "computational physics methods"): "Continuous models become finite algorithms with controllable error. Discretisation, stability, convergence and validation decide whether a simulation represents the physical system.",
    ("physics_branches", "nonlinear dynamics"): "Small changes can grow in nonlinear systems, producing chaos and sensitive dependence. Fixed points, limit cycles and bifurcations organise the long-term behaviour.",
    ("physics_branches", "geophysics"): "Earth is studied through waves, fields and deformation. Seismic tomography, geomagnetism and geodynamics infer inaccessible interior structure from surface and satellite measurements.",
    ("physics_branches", "biophysics"): "Physical constraints shape biological function across scales. Free energy, stochastic motion and mechanical forces explain molecular machines, membranes and signalling.",
    ("physics_branches", "quantum information"): "Qubits exploit superposition and entanglement to encode and process information. Measurement, decoherence and error correction determine what quantum devices can reliably compute.",
    ("physics_branches", "accelerator physics"): "Charged beams are guided by radio-frequency fields and magnets. Emittance, luminosity and beam stability set the reach of an experiment as much as peak energy does.",
    ("physics_branches", "detector physics"): "Detectors convert particle interactions into measurable signals. Ionisation, scintillation and readout chain response determine efficiency, resolution and background rejection.",
    ("physics_branches", "chemical physics"): "Chemical behaviour is explained with quantum structure and statistical dynamics. Potential energy surfaces, reaction pathways and solvent effects connect microscopic interactions to rates.",
    ("physics_branches", "atmospheric physics"): "Radiation, convection and rotation drive atmospheric motion. Radiative forcing and feedbacks determine energy balance, while turbulence limits deterministic prediction.",
    ("physics_branches", "medical physics"): "Imaging and therapy rely on controlled interactions of radiation with tissue. Dose, contrast, resolution and biological effect must be balanced for safe diagnosis and treatment.",
    ("physics_branches", "soft matter physics"): "Weak energies comparable to thermal fluctuations organise mesoscopic structure. Entropy, crowding and interfacial energy produce phases between simple liquids and solids.",
    ("physics_branches", "string theory"): "Particles are modelled as vibrating strings whose modes determine mass and charge. Extra dimensions and dualities connect seemingly different theories, but direct experimental tests remain difficult.",
    ("physics_branches", "quantum physics"): "States live in Hilbert space and observables are operators. Unitary evolution, superposition, entanglement and measurement give probabilities rather than deterministic trajectories.",
    ("physics_branches", "mathematical physics"): "Physical theories are made precise with equations, symmetries and function spaces. Existence, uniqueness and limiting behaviour clarify when a model is well posed and where approximations hold.",
    ("physics_branches", "acoustics"): "Sound is a pressure wave whose speed depends on the medium. Intensity, impedance, resonance and attenuation control how sources couple to listeners and detectors.",
    ("physics_branches", "nanophysics"): "At nanoscales, quantisation and surface effects dominate bulk intuition. Confinement changes spectra, tunnelling allows transport through barriers, and fabrication disorder strongly affects measurements.",
    ("physics_branches", "photonics"): "Photons are routed and controlled with refractive and semiconductor structures. Lasers, waveguides, modulators and detectors form systems whose bandwidth and loss set performance.",
    # Politics and Law
    ("politics_law", "federalism"): "Power is constitutionally divided between levels of government. Disputes over competence, funding and preemption define how federalism works in practice.",
    ("politics_law", "democracy"): "Democracy needs contested elections, participation and accountability. Majority rule is constrained by rights and by institutions that protect opposition.",
    ("politics_law", "monarchy"): "Monarchy combines symbolic continuity with varying formal powers. Constitutional monarchies differ sharply from absolute ones in where effective authority sits.",
    ("politics_law", "congress"): "Legislatures aggregate interests through committees and procedure. Agenda control and party discipline shape what reaches a vote.",
    ("politics_law", "executive"): "Executives implement law and direct administration. Emergency powers can expand authority, so oversight and review are central checks.",
    ("politics_law", "separation of powers"): "Institutions check one another rather than operating in sealed compartments. Practical governance depends on negotiated boundaries.",
    ("politics_law", "fundamental rights"): "Rights limit state action and protect core liberties. Scope, balancing tests and remedies vary by constitutional tradition.",
    ("politics_law", "civil law"): "Civil-law systems rely on codified rules and inquisitorial procedure. Judicial interpretation still develops the law between code revisions.",
    ("politics_law", "contract law"): "Contracts enable reliable exchange. Offer, acceptance, consideration and enforceability define duties, while good faith limits opportunism.",
    ("politics_law", "property law"): "Property is a bundle of rights rather than a single thing. Exclusion, use and transfer can be separated, and regulation balances private control with public interest.",
    ("politics_law", "human rights law"): "Human rights impose duties on states and increasingly on non-state actors. Treaties, courts and reporting create enforcement pathways.",
    ("politics_law", "political party"): "Parties aggregate interests and organise political power. Internal selection rules and funding shape whose voices count.",
    ("politics_law", "pressure group"): "Interest groups influence policy through advocacy, litigation and mobilisation. Access and resources create unequal influence.",
    ("politics_law", "bureaucracy"): "Bureaucracies implement policy through rules and discretion. Principal-agent problems and procedural rigidity are persistent challenges.",
    ("politics_law", "devolution"): "Devolution transfers powers without necessarily ending sovereignty. Fiscal autonomy and political consent determine durability.",
    ("politics_law", "liberalism"): "Liberalism prioritises individual liberty, rights and limited government. Variants differ over equality and the proper role of the state.",
    ("politics_law", "socialism politics"): "Socialism emphasises collective ownership and equality. Debates centre on markets, planning and democratic control of economic power.",
    ("politics_law", "capitalism politics"): "Capitalist politics centres on markets, property and profit. States create and continuously regulate the conditions that make markets possible.",
    ("politics_law", "environmentalism"): "Environmentalism links ecology, justice and policy. Precautionary and cost-benefit approaches frame different responses to uncertainty.",
    ("politics_law", "bicameralism"): "Two chambers can improve scrutiny and represent different interests. They can also produce gridlock when incentives diverge.",
    ("politics_law", "veto"): "A veto lets an actor block change. Its bargaining effect is often larger than the number of times it is actually used.",
    ("politics_law", "supreme court"): "A supreme court resolves final legal questions. Its legitimacy rests on reasoning, independence and the ability to secure compliance.",
    ("politics_law", "police"): "Policing combines prevention, investigation and order maintenance. Discretion, accountability and use of force are central issues.",
    ("politics_law", "media freedom"): "A free media enables scrutiny and pluralism. Ownership, regulation and journalist safety determine real independence.",
    ("politics_law", "governance"): "Governance includes formal institutions and informal networks. Effectiveness depends on legitimacy, capacity and accountability together.",
    # Psychology
    ("psychology", "cognitive behavioral therapy"): "CBT targets links among thoughts, feelings and behaviour. Cognitive restructuring, exposure and behavioural experiments are consolidated by between-session practice.",
    ("psychology", "operant conditioning"): "Behaviour is shaped by its consequences. Reinforcement schedules determine persistence, punishment has side effects, and extinction is context dependent.",
    ("psychology", "psychoanalysis"): "Psychoanalytic approaches emphasise unconscious conflict and early relationships. Transference and defence are central concepts; empirical support varies by claim.",
    ("psychology", "cognitive psychology"): "Mental processes are studied as information handling. Attention and working memory are limited, and models are tested with reaction times and error patterns.",
    ("psychology", "short term memory"): "Working memory holds limited information briefly and manipulates it. Chunking and rehearsal expand effective capacity; it is not a passive buffer.",
    ("psychology", "learning"): "Learning is a relatively stable change due to experience. Associative, observational and reinforcement-based mechanisms can all produce it.",
    ("psychology", "emotion"): "Emotion involves appraisal, physiology, expression and subjective feeling. Theories differ on sequencing, and regulation changes the trajectory.",
    ("psychology", "big five"): "The Big Five are broad trait dimensions identified factor-analytically. Scores are fairly stable but change with age and social roles.",
    ("psychology", "iq"): "IQ summarises performance on reasoning tests. It predicts some outcomes but is sensitive to test content, context and measurement error.",
    ("psychology", "attention"): "Attention selects and prioritises information. Capacity limits produce costs, and top-down goals compete with bottom-up salience.",
    ("psychology", "consciousness"): "Consciousness involves subjective experience and access to information. Levels, contents and neural correlates are separate questions.",
    ("psychology", "dream"): "Dreams occur across sleep stages with different features. Consolidation and threat-simulation theories explain some patterns but not all.",
    ("psychology", "conformity"): "Conformity reflects informational and normative influence. Group size, unanimity and task difficulty moderate the effect.",
    ("psychology", "group dynamics"): "Groups develop norms, roles and cohesion. Process losses and polarisation can override individual judgement.",
    ("psychology", "aggression"): "Aggression is behaviour intended to harm. Hostile and instrumental forms differ; situational cues interact with learning and disposition.",
    ("psychology", "developmental psychology"): "Development spans cognitive, social and emotional change. Nature and nurture interact through experience-dependent processes.",
    ("psychology", "erikson stages"): "Erikson proposed psychosocial crises across the lifespan. Each stage frames a central tension, but outcomes are not rigidly fixed.",
    ("psychology", "abnormal psychology"): "Disorders are identified by patterns of distress and impairment. Categorical and dimensional models each have strengths and limitations.",
    ("psychology", "anxiety"): "Anxiety combines threat appraisal with physiological arousal. Avoidance maintains it; exposure and cognitive change reduce it.",
    ("psychology", "ocd"): "Obsessions are intrusive thoughts; compulsions are repeated acts that reduce distress. Habituation and response prevention are key mechanisms of change.",
    ("psychology", "bipolar disorder"): "Bipolar disorder involves alternating manic and depressive episodes. Sleep, substance use and treatment adherence influence the course.",
    ("psychology", "addiction"): "Addiction involves compulsive use despite harm. Reinforcement, tolerance, withdrawal and environmental cues interact.",
    ("psychology", "coping"): "Coping strategies manage stressful demands. Problem-focused and emotion-focused approaches fit different contexts, so flexibility matters.",
    ("psychology", "therapy"): "Psychotherapies use structured methods to change thoughts, emotions or behaviour. The therapeutic alliance and mechanism-specific techniques both contribute.",
    ("psychology", "experiment"): "Experiments manipulate an independent variable to test causality. Randomisation, control groups and measurement validity determine what can be concluded.",
}


def _key(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(title or "").lower())).strip()


def _looks_like_formula(text: str) -> bool:
    return bool(re.search(r"[=≈→<>∫√^]|(?:\d)", text))


def is_advanced_request(question: str) -> bool:
    """True when the question explicitly asks for deeper treatment."""
    return bool(ADVANCED_REQUEST.search(str(question or "")))


# Applied questions such as "Explain pH of 0.01 M HCl at an advanced level" do
# not score as a definition card, but their deeper explanation still belongs to
# the matching concept card. Add narrow hints instead of guessing a subject.
SOLVER_CARD_HINTS = (
    (re.compile(r"\bpH\b", re.I), "chemistry", "pH"),
    (re.compile(r"\bideal gas\b", re.I), "physics", "Ideal gas law"),
    (re.compile(r"\bohm'?s law\b|\bresistor\b", re.I), "physics", "Ohm's law"),
    (re.compile(r"\bfriction\b", re.I), "physics", "Friction"),
    (re.compile(r"\bprojectile\b", re.I), "physics", "Projectile motion"),
    (re.compile(r"\bdoppler\b", re.I), "physics", "Doppler effect"),
    (re.compile(r"\bde broglie\b", re.I), "physics", "Quantum mechanics"),
    (re.compile(r"\brest energy\b|\bmass[-\s]energy\b", re.I), "physics", "Relativity"),
    (re.compile(r"\bhalf[-\s]life\b", re.I), "physics", "Nuclear fission and fusion"),
    (re.compile(r"\bdebye length\b", re.I), "physics_branches", "Plasma Physics"),
    (re.compile(r"\bpoisson'?s ratio\b", re.I), "physics_branches", "Geophysics"),
    (
        re.compile(
            r"\bbernoulli(?:'s|s)?\s+(?:equation|principle|theorem)\b"
            r"|\bbernoulli\s+flow\b"
            r"|\bpressure drop\b.*\bpipe\b",
            re.I,
        ),
        "physics_branches",
        "Fluid Dynamics",
    ),
    (re.compile(r"\bnernst\b", re.I), "physics_branches", "Biophysics"),
    (re.compile(r"\blyapunov\b", re.I), "physics_branches", "Nonlinear Dynamics"),
    (re.compile(r"\bmonte carlo\b", re.I), "physics_branches", "Computational Physics Methods"),
    (re.compile(r"\bheat equation\b|\bfinite[-\s]difference\b|\bcfl\b", re.I), "physics_branches", "Computational Physics Methods"),
    (re.compile(r"\binfinite (?:square )?well\b", re.I), "physics_branches", "Quantum Physics"),
    (re.compile(r"\bgrating\b|\bdiffraction\b", re.I), "physics_branches", "Optics"),
    (re.compile(r"\bsnell\b|\brefraction\b", re.I), "physics_branches", "Optics"),
    (re.compile(r"\bblack hole\b|\bschwarzschild\b", re.I), "physics_branches", "General Relativity"),
    (re.compile(r"\bredshift\b|\bhubble\b", re.I), "physics_branches", "Cosmology"),
    (re.compile(r"\bspin[-\s]?(?:1/2|one[-\s]half)\b", re.I), "physics_branches", "Quantum Information"),
    (re.compile(r"\blength contraction\b|\bvelocity addition\b", re.I), "physics", "Relativity"),
)


def hint_for(question: str) -> Optional[Tuple[str, str]]:
    """Return (subject key, card title) for a narrow applied-topic hint."""
    text = str(question or "")
    for pattern, subject_key, title in SOLVER_CARD_HINTS:
        if pattern.search(text):
            return subject_key, title
    return None


def build(card: "TopicCard", related: Sequence["TopicCard"]) -> str:
    """Build a structured advanced answer from one card and its subject lens."""
    lens = SUBJECT_LENSES.get(card.subject_key, DEFAULT_LENS)
    note = ADVANCED_NOTES.get((card.subject_key, _key(card.title)), "")

    formula = str(card.formula or "").strip()
    if formula and _looks_like_formula(formula):
        formal = f"{formula}."
    else:
        # A comma-separated list is a reminder, not an equation. Use the
        # subject lens instead of pretending it is a formula.
        formal = lens["formal"]

    if note:
        mechanism = note
    else:
        mechanism = (
            f"{lens['mechanism']} For {card.title.lower()}, connect the stated "
            "structure to the behaviour it produces."
        )

    main_points = str(card.points or "").strip().rstrip(".")
    sections = [
        f"{card.title} - advanced explanation",
        f"Definition: {card.summary}",
        f"Formal core: {formal}",
        f"Mechanism: {mechanism}",
        f"Advanced framing: {lens['frame']}",
        f"Main points: {main_points}. {lens['limits']}",
    ]
    text = "\n".join(sections)
    if related:
        follow_up = related[0]
        text += f"\nRelated: {follow_up.title} - {follow_up.summary}"
    return text


__all__ = ["ADVANCED_REQUEST", "build", "hint_for", "is_advanced_request"]
