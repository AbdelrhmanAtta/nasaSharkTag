# Maritime Orbiters
### Project Sentinel: You cannot protect what you cannot see. Our mission is to make the invisible, visible.
Every year, an estimated 80 million sharks are killed, pushing ocean ecosystems toward collapse. Current conservation efforts are failing because they rely on static, "blind" protected areas for animals that are constantly migrating.
Project Sentinel is our answer. It's a predictive intelligence platform that fuses NASA satellite data (PACE & SWOT) with a AI-powered smart-tag to create a global, real-time map of shark behavior. Our system doesn't just show where sharks are, it predicts when they will hunt, enabling the creation of proactive, Dynamic Marine Protected Areas that can finally keep pace with the animals they are designed to save.

[Maritime Orbiter's website](https://your-url-goes-here.com) 

![Project Sentinel](https://private-user-images.githubusercontent.com/228541521/497376522-ffd052e5-89c9-4258-ba7c-790bab6493ec.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTk1NDAxNDMsIm5iZiI6MTc1OTUzOTg0MywicGF0aCI6Ii8yMjg1NDE1MjEvNDk3Mzc2NTIyLWZmZDA1MmU1LTg5YzktNDI1OC1iYTdjLTc5MGJhYjY0OTNlYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTAwNFQwMTA0MDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wYWYwNWYyODg5YTEwNWFhODA1MDM4ZjA1ZjcxZTcyOTAzZWRkM2JmNGNiNjg5YjY5MzFjYWI2NGFhNmVmYjRiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.8i87yePMyjt1M8HvVmguRcaU-XtIfhQ6iWJXrX9U_kc)

## Table Of Contents
1. [**Problem**](#the-problem)
2. [**Features**](#features)
3. [**TAG device**](#tag)
4. [**Predictive AI**](#sentinels-ai)
5. [**Web Platform**](#web-platform)
6. [**Our Team**](#our-team-maritime-orbiters)
7. [**Resources**](#resources)
### The Problem
A silent annihilation is unfolding beneath the waves, and our mission is born from the devastating reality of the numbers. In the last 50 years, the global population of oceanic sharks and rays has plummeted by a catastrophic 71% (Pacoureau et al., 2021). This isn't a natural decline; it's a slaughter driven by an industrial scale of overfishing that kills an estimated 80 million sharks annually (Cardeñosa et al., 2023).

This is a global cascade failure. As apex predators, sharks are the guardians of the ocean, maintaining the delicate balance of the food web and even helping to fight climate change by protecting "blue carbon" sinks like seagrass meadows. Their removal triggers a domino effect of destruction that threatens global fish stocks that feed billions of people.

The tragic irony is that our efforts to protect them are failing. We rely on static Marine Protected Areas, fixed boxes on a map, to protect animals that migrate thousands of kilometers. We are trying to build a sanctuary in the path of a hurricane. This is the critical conservation gap we are here to solve. We refuse to let the guardians of our ocean vanish simply because we cannot see them.
### Features
Project Sentinel is a comprehensive, end-to-end solution designed to turn the tide. Our platform is built on a foundation of powerful, interconnected features that transform raw data into actionable conservation.

* Global Predictive Map: Our core feature is a dynamic, interactive world map that doesn't just show where sharks are, but uses AI to forecast where they are most likely to be next and their eating habits along with other data collected from the TAG.

AI-Powered Foraging Score: We've developed a Random Forest Regressor model that analyzes environmental data to generate a real-time "Foraging Probability Score" from 0.0 to 1.0 for any given ocean region.

Dynamic Sanctuary Concept: Our platform demonstrates how this predictive data can be used to create Dynamic Marine Protected Areas—intelligent, mobile sanctuaries that adapt and move with the animals they are designed to protect.

Smart Tag Design: We've conceptualized a next-generation shark tag that uses an adaptive Bayesian Markov chain model to make intelligent decisions about when to transmit data, radically increasing battery life and enabling long-term tracking missions.

Public Engagement Portal: To win the fight for our oceans, we must win hearts and minds. Our platform includes a dedicated educational portal with an interactive quiz, an e-book, and personal "Shark Tales" to transform public fear into fascination and advocacy.
### TAG
The foundation of our ground-truth data is our innovative smart tag—a device conceptualized to be more intelligent, resilient, and energy-efficient than anything available today.

Intelligent Power Management: The tag's greatest innovation is its power-saving system. Instead of constant, battery-draining data streams, it uses an adaptive Bayesian Markov chain model. This onboard TinyML model allows the tag to learn the shark's patterns and make smart decisions, only transmitting data during key events or surfacing periods, extending mission life from months to potentially years.

Rich Sensor Suite: The tag is equipped with a high-fidelity accelerometer and gyroscope to capture the subtle movements that define a shark's behavior—the steady glide of travel, the languid motion of rest, and the explosive acceleration of a hunt.

Real-Time Satellite Uplink: When the tag surfaces, it connects to satellite networks to upload its compressed data packets to our cloud platform, feeding the AI engine with a constant stream of new information.

Rugged & Non-Invasive Design: The conceptual design is hydrodynamic, rugged, and affixed non-invasively to the dorsal fin to ensure the shark's welfare is the top priority. We cannot protect these animals by harming them in the process.
### Sentinel's AI
Our AI is the brain of the operation, a dual-model engine that works in synergy to translate complex data into a simple, powerful prediction.

The Classifier - "What is the shark doing?": The first stage is a Random Forest Classifier. It is trained on our synthetic tag data to analyze motion patterns. Its sole job is to classify the shark's current behavior into discrete states: ‘Traveling,’ ‘Resting,’ or the crucial ‘Hunting’ state.

The Regressor - "How good is this hunting spot?": When the Classifier flags a ‘Hunting’ event, our Random Forest Regressor activates. It fuses the location of the hunt with multi-layered NASA satellite data, Chlorophyll, a concentrations from PACE and sea surface anomalies from SWOT. Its output is the "Foraging Probability Score," a number from 0.0 to 1.0 that tells conservationists how valuable that specific area is as a feeding ground.

Simulation & Validation: To prove this concept within the hackathon, we used ChatGPT to generate a robust, synthetic dataset simulating a year in the life of a shark. This "digital twin" allowed us to successfully train, test, and validate our AI pipeline from end to end.
### Web Platform
Our web platform is the bridge between our powerful AI and the people who can make a difference. It’s divided into two key areas: the data-rich Dashboard and the story-driven Engagement Portal.

The Sentinel Dashboard: This is the command center for scientists and policymakers.  It features an interactive 3D globe where users can visualize real-time simulated shark tracks, view the predictive foraging heatmaps generated by our AI, and explore the layers of NASA satellite data that feed our models. It's designed to make complex data intuitive, beautiful, and actionable. For real-life use, an ID is needed to view a designated shark, one of the many sharks that are tagged and are monitroed across the oceans.

###### The Public Engagement Portal: This is where our passion for changing perceptions comes to life.

The Guardian's Quiz: A fun, 10-question quiz designed to bust common myths and teach fascinating truths about sharks.

Shark Tales: We believe empathy begins with a story. This section presents beautifully written narratives about different shark species, allowing users to connect with them as individuals, not monsters.

Mission and motivation:

Vision:
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

AI/ML: Python, Scikit-learn, Pandas

Backend: Flask

Frontend: React, Mapbox

###### Generative AI:

Gemini 2.5 Pro: Assisted with narrative, storytelling, and documentation.

ChatGPT: Assisted with code refinement and the critical generation of our synthetic shark tag dataset.

###### Scientific Data Sources:

Cardeñosa, D., et al. (2023). CITES listing of sharks and rays... Science.

Pacoureau, N., et al. (2021). Half a century of global decline... Nature.
