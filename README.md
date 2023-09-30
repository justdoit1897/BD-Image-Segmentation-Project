# 🇮🇹 DESCRIZIONE

Nato come parte dell'esame del corso di **Big Data** presso **l'Università degli Studi di Palermo**, questo progetto parte da una competizione lanciata dallo **UW-Madison Carbone Cancer Center**, il cui obiettivo prevedeva l'implementazione di un modello per **segmentare automaticamente stomaco e intestini** all'interno di scansioni MRI del tratto gastrointestinale. 
L'uso di un modello di deep learning è da intendersi come **strumento di supporto** all'attività del personale medico, con l'effetto positivo di velocizzare le procedure decisionali relative alla cura dei pazienti (dato che le radiazioni devono essere somministrate in modo accurato e coerente con lo stato della malattia).
Il **dataset** si compone di **scansioni in bianco e nero** del tratto gastrointestinale e da **alcune segmentazioni** (effettuate manualmente dal personale medico) codificate in RLE, opportunamente suddivise per caso clinico e giornata di osservazione. Un obiettivo del modello dev'essere anche quello di **generalizzare su casi parzialmente visti** o con disturbi di varia natura.
Le metriche usate per la valutazione sono il **coefficiente di Dice** e la **distanza di Hausdorff**, utilizzate sia in versione 2D che in versione 3D (trattando le scansioni come volumi).

L'intera pipeline per i big data è discussa nel dettaglio all'interno di **notebook python** (file `.ipynb`), potendo osservare, passo passo, l'elaborazione del dataset e la creazione del modello, avendo coscienza delle ratio dietro le scelte progettuali.

# 🏴󠁧󠁢󠁥󠁮󠁧󠁿 DESCRIPTION

At first intended as a practical exam for **Università degli Studi di Palermo's Big Data** course, this project deals with a competition issued by **UW-Madison Carbone Cancer Center** whose objective is the definition of a model able to **automatically segment stomach, large bowel and small bowel** inside a MRI scan.
The use of a deep learning model has to be intended as a **support to medical staff's work**, by speeding up the decisional process about patients' health (since the radiation quantity to be submitted has to be carefully calibrated and coherent with the cancer's stage).
The **dataset** is composed of **B/W scans of gastrointestinal tract** and of some **manually-crafted** (by the medical staff) **segmentations**, encoded as RLE strings, having everything divided into clinical cases and days of observation. A model's objective is to be able to **generalize** correctly on **partially-seen cases** or on **disturbed images **(e.g. noisy images).
The metrics used for the model's evaluation are **Dice's coefficient** and **Haudorff's distance**, both implemented in their 2D and 3D forms (when dealing with volumes).

The whole big data pipeline is carefully discussed into **python notebooks** (`.ipynb` files), which allow one to understend, step by step, how the dataset was used and what were the reasons behind the creation of the model.

*Developed By **Vincenzo Fardella** & **Mario Tortorici** @ **Università degli Studi di Palermo***
