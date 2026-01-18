# Scheduling-Charging-Sessions-for-EV-using-a-Multi-Agent-Sequential-Game-driven-by-RL

Dieses System implementiert einen zweistufigen Entscheidungsrahmen für die Planung von Elektrofahrzeugen auf der Grundlage der Spieltheorie:
Planung der oberen Ebene: Preisentscheidungen mit gemischter Strategie auf der Grundlage sequenzieller Spiele
Planung der unteren Ebene: Clustered Charging Schedules auf der Grundlage von Stackelberg-Spielen

This electric vehicle intelligent dispatching system adopts a two-level game-theoretic framework, consisting of an upper-level sequential game–based scheduling module and a lower-level Stackelberg game–based scheduling module. A full run of the system requires approximately 2.5–4.5 hours.

Before execution, please ensure that the environment meets the following requirements: Python 3.8 or above, at least 8 GB of RAM, and no less than 5 GB of available storage space. It is recommended to use the specified versions of PyTorch, NumPy, and other dependency packages to avoid compatibility issues. During execution, the system will automatically create multiple result directories; please make sure that write permissions are enabled. If memory limitations occur, the batch size or number of training epochs can be reduced. In the event of CUDA-related errors, please verify the compatibility between the installed PyTorch and CUDA versions.

During training, do not arbitrarily modify the core network architecture or the reward function. For academic use, please cite this work in accordance with the relevant guidelines. It is recommended to conduct small-scale tests before a full run. For troubleshooting, please record complete error messages and provide them for further analysis.
