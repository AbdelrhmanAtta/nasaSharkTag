# Maritime Orbiters
### Project Sentinel: You cannot protect what you cannot see. Our mission is to make the invisible, visible.
Every year, an estimated 80 million sharks are killed, pushing ocean ecosystems toward collapse. Current conservation efforts are failing because they rely on static, "blind" protected areas for animals that are constantly migrating.
Project Sentinel is our answer. It's a predictive intelligence platform that fuses NASA satellite data (PACE & SWOT) with a AI-powered smart-tag to create a global, real-time map of shark behavior. Our system doesn't just show where sharks are, it predicts when they will hunt, enabling the creation of proactive, Dynamic Marine Protected Areas that can finally keep pace with the animals they are designed to save.

[Maritime Orbiter's website - not deployed](https://github.com/AbdelrhmanAtta/nasaSharkTag/tree/main/webpage) 

![Project Sentinel](https://github.com/AbdelrhmanAtta/nasaSharkTag/blob/main/assets/1.png?raw=true)

## Table Of Contents
1. [**Problem**](#the-problem)
2. [**Features**](#features)
3. [**Mathematical framework**](#mathematical-framework)
4. [**TAG device**](#tag)
5. [**Predictive AI**](#sentinels-ai)
6. [**Web Platform**](#web-platform)
7. [**Our Team**](#our-team-maritime-orbiters)
8. [**Resources**](#resources)
9. [**Future of the Sentinel**](#future-of-the-sentinel)
### The Problem
A silent annihilation is unfolding beneath the waves, and our mission is born from the devastating reality of the numbers. In the last 50 years, the global population of oceanic sharks and rays has plummeted by a catastrophic 71% (Pacoureau et al., 2021). This isn't a natural decline; it's a slaughter driven by an industrial scale of overfishing that kills an estimated 80 million sharks annually (Cardeñosa et al., 2023).

This is a global cascade failure. As apex predators, sharks are the guardians of the ocean, maintaining the delicate balance of the food web and even helping to fight climate change by protecting "blue carbon" sinks like seagrass meadows. Their removal triggers a domino effect of destruction that threatens global fish stocks that feed billions of people.

The tragic irony is that our efforts to protect them are failing. We rely on static Marine Protected Areas, fixed boxes on a map, to protect animals that migrate thousands of kilometers. We are trying to build a sanctuary in the path of a hurricane. This is the critical conservation gap we are here to solve. We refuse to let the guardians of our ocean vanish simply because we cannot see them.
### Features
Project Sentinel is a comprehensive, end-to-end solution designed to turn the tide. Our platform is built on a foundation of powerful, interconnected features that transform raw data into actionable conservation.

* Global Predictive Map: Our core feature is a dynamic, interactive world map that doesn't just show where sharks are, but uses AI to forecast where they are most likely to be next and their eating habits along with other data collected from the TAG.

* AI-Powered Foraging Score: We've developed a Random Forest Regressor model that analyzes environmental data to generate a real-time "Foraging Probability Score" from 0.0 to 1.0 for any given ocean region.

* Dynamic Sanctuary Concept: Our platform demonstrates how this predictive data can be used to create Dynamic Marine Protected Areas—intelligent, mobile sanctuaries that adapt and move with the animals they are designed to protect.

* Smart Tag Design: We've conceptualized a next-generation shark tag that uses an adaptive Bayesian Markov chain model to make intelligent decisions about when to transmit data, radically increasing battery life and enabling long-term tracking missions.

* Public Engagement Portal: To win the fight for our oceans, we must win hearts and minds. Our platform includes a dedicated educational portal with an interactive quiz, an e-book, and personal "Shark Tales" to transform public fear into fascination and advocacy.
### Mathematical framework
Our framework is a geospatial data fusion pipeline that uses machine learning to identify environmental niches likely to be shark foraging hotspots. The workflow is as follows:

1. Data Fusion: We fuse multi-source NASA satellite data (MODIS for Chlorophyll/Temperature and SWOT for Sea Surface Height) into a single, unified analysis grid. This is achieved using a cKDTree algorithm for efficient nearest-neighbor interpolation, solving the challenge of mismatched data resolutions.

2. Niche Definition: We mathematically define an ideal foraging hotspot based on a combination of environmental conditions (e.g., high chlorophyll, specific temperature range). This creates a "perfect" hotspot signature that we use to train our model.

3. Predictive Classification: A Random Forest Classifier is trained to recognize this environmental signature. The model then analyzes the entire fused data grid and calculates the probability for each pixel, generating a final heatmap of likely foraging locations.

4. Tag Efficiency (Markov Model): Separately, the smart tag itself is designed with a Markov model. This allows the tag to intelligently decide when sensor readings indicate a significant event, enabling it to transmit data only when necessary to maximize battery life.
### TAG
Our primary innovation is the development of the first conceptualized, non-invasive smart tag that integrates a comprehensive sensor suite with next-generation satellite communication in an exceptionally small form factor. Unlike bulky or invasive alternatives, the Sentinel Tag is designed to provide rich, multi-layered data—including motion (IMU), temperature, pressure (depth), and precise location—while prioritizing animal welfare. Its core novelty lies in its planned integration of the Argos satellite system, moving beyond the limitations of coastal GSM networks to offer true, global open-ocean tracking capabilities. This represents a significant leap forward in creating a viable, long-term, and humane tool for marine research.

This innovation is brought to life through the following key systems:

Intelligent Power Management
The tag's greatest innovation is its power-saving system. Instead of constant, battery-draining data streams, it uses an adaptive Bayesian Markov chain model. This onboard TinyML model allows the tag to learn the shark's patterns and make smart decisions, only transmitting data when significant and noticeable changes occur, extending mission life from months to potentially years.

Rich Sensor Suite
The tag is equipped with a high-fidelity accelerometer and gyroscope (IMU) to capture the subtle movements that define a shark's behavior—the steady glide of travel, the languid motion of rest, and the explosive acceleration of a hunt. This is complemented by onboard temperature and pressure sensors to provide crucial environmental context.

Real-Time Intelligent Satellite Uplink
When the tag surfaces, it connects to satellite networks to upload its compressed data packets to our cloud platform. Furthermore, when the shark is at great depths and satellite connection is lost, the tag stores sensor data in a buffer with timestamps. When the shark resurfaces and connection is restored, this buffered data is transmitted. The location data lost during the deep dive is compensated for by our AI's predictive path model, ensuring a more complete dataset.

Rugged & Non-Invasive Design
The conceptual design is hydrodynamic, rugged, and affixed non-invasively to the dorsal fin to ensure the shark's welfare is the top priority. We cannot protect these animals by harming them in the process.

The current proof-of-concept measures approximately 4×9 cm and demonstrates these key innovations. The final version will be smaller, feature a longer-lasting battery, improved accuracy, robust error handling, and will be fully Argos-enabled for seamless global tracking.
![TAG hardware](https://github.com/AbdelrhmanAtta/nasaSharkTag/blob/main/assets/TAG.jpg?raw=true)
### Sentinel's AI
Our AI is the brain of the operation, a dual-model engine that works in synergy to translate complex data into a simple, powerful prediction.

The Classifier - "What is the shark doing?": The first stage is a Random Forest Classifier. It is trained on our synthetic tag data and NASA satellite data to analyze motion patterns. Its sole job is to classify the shark's current behavior into discrete states: ‘Traveling,’ ‘Resting,’ or the crucial ‘Hunting’ state.

![AI-NASA](https://raw.githubusercontent.com/AbdelrhmanAtta/nasaSharkTag/refs/heads/main/assets/AI%20predict%20satellite.jpg?raw=true)
The Regressor - "How good is this hunting spot?": When the Classifier flags a ‘Hunting’ event, our Random Forest Regressor activates. It fuses the location of the hunt with multi-layered NASA satellite data, Chlorophyll, a concentrations from PACE and sea surface anomalies from SWOT. Its output is the "Foraging Probability Score," a number from 0.0 to 1.0 that tells conservationists how valuable that specific area is as a feeding ground.

![AI-tag](https://raw.githubusercontent.com/AbdelrhmanAtta/nasaSharkTag/refs/heads/main/assets/AI%20predict%20tag.jpg?raw=true)
Simulation & Validation: To prove this concept within the hackathon, we used ChatGPT to generate a robust, synthetic dataset simulating a year in the life of a shark. This "digital twin" allowed us to successfully train, test, and validate our AI pipeline from end to end.
### Web Platform
Our web platform is the bridge between our powerful AI and the people who can make a difference. It’s divided into two key areas: the data-rich Dashboard and the story-driven Engagement Portal.
![Home](https://raw.githubusercontent.com/AbdelrhmanAtta/nasaSharkTag/refs/heads/main/assets/Home.png?raw=true)
The Sentinel Dashboard: This is the command center for scientists and policymakers.  It features an interactive 3D globe where users can visualize real-time simulated shark tracks, view the predictive foraging heatmaps generated by our AI, and explore the layers of NASA satellite data that feed our models. It's designed to make complex data intuitive, beautiful, and actionable. For real-life use, an ID is needed to view a designated shark, one of the many sharks that are tagged and are monitroed across the oceans.
![Global 3D map](https://raw.githubusercontent.com/AbdelrhmanAtta/nasaSharkTag/refs/heads/main/assets/Global%20map%20web.png?raw=true)

##### The Public Engagement Portal: This is where our passion for changing perceptions comes to life.

The Guardian's Quiz: A fun, 10-question quiz designed to bust common myths and teach fascinating truths about sharks.

Shark Tales: We believe empathy begins with a story. This section presents beautifully written narratives about different shark species, allowing users to connect with them as individuals, not monsters.

Mission and motivation: Our mission is to close the gap between human ignorance and the urgent need for action. We build tools to make the invisible guardians of our ocean visible, transforming a failing, reactive conservation effort into a proactive, data-driven science.

Vision: We envision a future where technology and nature are in harmony, where guesswork is eliminated from conservation, and where a global community of citizen-guardians feels the pulse of the planet and is empowered, daily, to protect it.
### Our Team: Maritime Orbiters
We are a passionate, multidisciplinary team brought together by a shared sense of urgency to protect our oceans. We believe that technology, when guided by purpose, can solve our planet's greatest challenges.
* Team Leader - [Marwan Mohamed]: Oversees the project, organizes activity, synchronizes milestones with conservation goals, and documents the project.
* Embedded IoT & Hardware – [Abdelrhman Atta]: Codes and designs shark tags, implements TinyML, and makes them rugged and battery-efficient.
* AI – [Lojine Ahmed] [Ali Gad] [Mariam Ramy]: Utilise AI models to simulate the sharks' behavior and predict their future movement based on cloud data and tag data.
* Web – [Ahmed Abozeid]: Manages cloud infrastructure, data storage, visualization, and web interface for researchers.
### Resources
Our project stands on the shoulders of giants; from NASA's incredible open-source data to the cutting-edge tools that enabled our small team to build something big.

###### NASA Data:

PACE Mission Data: Used for identifying Chlorophyll-a concentrations as a proxy for phytoplankton, the base of the ocean food web.

SWOT Mission Data: Used to identify sea surface height anomalies, indicating ocean eddies and fronts where nutrients and prey congregate.

###### Core Technologies:

AI/ML: Python, Scikit-learn, Pandas.

Backend: ASP.NET Core MVC, MS SQL, MQTTServiceConnection.

Frontend: HTML, javascript, CSS, Bootstrap.

###### Generative AI:

Gemini 2.5 Pro: Assisted with narrative, storytelling, and documentation.

ChatGPT: Assisted with code refinement and the critical generation of our synthetic shark tag dataset.

###### Scientific Data Sources:

Cardeñosa, D., Shea, O., Feldheim, K., Heithaus, M. R., & Chapman, D. D. (2023). CITES listing of sharks and rays has had limited impact on their international trade. Science, 379(6633), 704-709.
* [Read more](https://www.science.org/doi/10.1126/science.adf6423)

Pacoureau, N., Rigby, C. L., Kyne, P. M., Finucci, B., Jabado, R. W., ... & Dulvy, N. K. (2021). Half a century of global decline in oceanic sharks and rays. Nature, 589(7843), 567–571.
* [Read more](https://www.nature.com/articles/s41586-020-03173-9) 

### Future of the Sentinel
### Sentinel's Tag
The device built during the hackathon served as a powerful proof-of-concept. The following roadmap details our plan to evolve it into a mission-ready, field-deployable device.

Onboard Intelligence: The Leap to Edge AI
To elevate the tag from a simple data logger to a truly intelligent agent, our roadmap includes the implementation of Edge AI directly on the device's MCU. This will enable advanced, real-time decision-making:

* Predictive Dead Reckoning: For periods when the shark is too deep for a GPS fix, the Edge AI will use IMU data to predict and fill in location gaps, providing a more continuous data stream.

* Intelligent Power Management: The onboard Markov chain model will be enhanced to actively manage power consumption, calculating the optimal times to transmit data based on battery level, behavior, and satellite availability.

* Onboard Diagnostics: The AI will run a continuous warning system to detect sensor anomalies or physical damage, ensuring data integrity.

Next-Generation Hardware & Miniaturization
A key goal is to drastically reduce the tag's physical footprint to minimize its impact on the animal.

* Next PCB: Our current prototype is approximately 4x9 cm. The next design iteration will target a miniaturized footprint of around 2.5x7.5 cm.

* Component Evolution: This miniaturization and performance upgrade is made possible by transitioning to next-generation components.
| Component | Hackathon Prototype | Next-Generation Design |
| :--- | :--- | :--- |
| **MCU** | STM32F4 Series | **STM32U5 Series (Ultra-Low-Power)** |
| **Comms**| SIM800L (2G GSM) & NEO-7M (GPS) | **Arribada Argos SMD Module (Global Satellite)** |
| **Antenna** | Generic | **Meteor FW.43 Flexible Whip (Bite-Proof)** |
| **IMU** | MPU6050 | **TDK InvenSense ICM-45686 (High-Precision)** |
| **Battery** | Standard Li-ion 3.7V | **ER18505 (High-Capacity, Long-Life)** |
Firmware & Power Management Overhaul
The firmware will be completely re-architected for maximum efficiency, with a goal of achieving a battery life of up to 3 years.

* Interrupt-Based Architecture: The system will move from a polling-based model to being 100% interrupt-based. The device will remain in a deep sleep state, consuming near-zero power, and will only wake when a sensor interrupt reports a significant event.

* Multi-Modal Operation: The tag will feature several distinct operating modes, such as:

** Working Mode: Normal data collection and transmission.

** Low Power Mode: Drastically reduced sensing and transmission frequency.

** Find Me Mode: A special transmission mode to aid in device recovery.

* Robust Error Handling: The new firmware will include advanced error handling to manage data corruption, transmission failures, and sensor malfunctions gracefully.
### Sentinel's AI

