# OVERVIEW

## DESCRIZIONE

Nel 2019, si stima che in tutto il mondo siano state diagnosticate 5 milioni di persone con un tumore del tratto gastrointestinale. Di questi pazienti, circa la metà può essere sottoposta a radioterapia, di solito somministrata per 10-15 minuti al giorno per 1-6 settimane. Gli oncologi radioterapisti cercano di somministrare dosi elevate di radiazioni utilizzando fasci di raggi X puntati sui tumori, evitando lo stomaco e l'intestino. Con le nuove tecnologie, come i sistemi integrati di risonanza magnetica e acceleratore lineare, noti anche come MR-Linac, gli oncologi sono in grado di visualizzare la posizione giornaliera del tumore e dell'intestino, che può variare di giorno in giorno. In queste scansioni, gli oncologi radioterapisti devono delineare manualmente la posizione dello stomaco e dell'intestino per regolare la direzione dei fasci di raggi X in modo da aumentare la dose erogata al tumore ed evitare lo stomaco e l'intestino. Si tratta di un processo lungo e laborioso che può prolungare i trattamenti da 15 minuti al giorno a un'ora al giorno, il che può essere difficile da tollerare per i pazienti, a meno che il deep learning non aiuti ad automatizzare il processo di segmentazione. Un metodo per segmentare lo stomaco e l'intestino renderebbe i trattamenti molto più veloci e consentirebbe a un maggior numero di pazienti di ricevere un trattamento più efficace.

L'UW-Madison Carbone Cancer Center è un pioniere della radioterapia basata sulla risonanza magnetica e dal 2015 tratta i pazienti con radioterapia guidata dalla risonanza magnetica in base alla loro anatomia quotidiana. L'UW-Madison ha generosamente accettato di sostenere questo progetto che fornisce risonanze magnetiche anonime di pazienti trattati presso l'UW-Madison Carbone Cancer Center. L'Università del Wisconsin-Madison è un'università pubblica di ricerca con sede a Madison, nel Wisconsin. L'Idea del Wisconsin è l'impegno dell'università nei confronti dello Stato, della nazione e del mondo, affinché i suoi sforzi vadano a beneficio di tutti i cittadini.

In questa competizione, dovrete creare un modello per segmentare automaticamente lo stomaco e l'intestino nelle scansioni MRI. Le scansioni MRI provengono da pazienti oncologici reali che hanno effettuato 1-5 scansioni MRI in giorni diversi durante il loro trattamento con radiazioni. Dovrete basare il vostro algoritmo su un data set di queste scansioni per trovare soluzioni creative di deep learning che aiutino i pazienti oncologici a ricevere cure migliori.

![Segmentazione scansione](https://lh5.googleusercontent.com/zbBUgbj1jyZxyu3r1vr5zKKr8yK1hSdwAM3HpD_n6j2W-5-wKP3ZRusi_3yskSgnC-tMRKqOEtLycbLkTWCJAUe4Cylv_VsW81DYI4ray02uZLeSnlzAuZRIU7L2Q0KURYSMqFI)

In questa figura, il tumore (linea spessa rosa) è vicino allo stomaco (linea spessa rossa). Al tumore vengono indirizzate dosi elevate di radiazioni, evitando lo stomaco. I livelli di dose sono rappresentati dall'arcobaleno dei contorni, con le dosi più alte rappresentate dal rosso e quelle più basse dal verde.

Il cancro è già abbastanza pesante. Se riuscirete nell'intento, permetterete agli oncologi radioterapisti di somministrare in modo sicuro dosi più elevate di radiazioni ai tumori, evitando lo stomaco e l'intestino. Questo renderà più veloci i trattamenti quotidiani dei pazienti oncologici e consentirà loro di ottenere cure più efficaci con meno effetti collaterali e un migliore controllo del cancro a lungo termine.

## EVALUATION

Questa competizione viene valutata in base al Dice coefficient e a 3D Hausdorff distance. Il coefficiente Dice può essere utilizzato per confrontare il rapporto tra i pixel di una segmentazione prevista e la corrispondente verità a terra. La formula è data da:

$$2\cdot \frac{|X\cap Y|}{|X|+|Y|}$$

dove X è l'insieme dei pixel previsti e Y è la ground truth. Il coefficiente Dice è pari a 0 quando sia X che Y sono vuoti. Il punteggio della classifica è la media dei coefficienti di Dice per ogni immagine del test set.
La distanza di Hausdorff è un metodo per calcolare la distanza tra gli oggetti di segmentazione A e B, considerando il punto più lontano di A dal punto più vicino di B.
Per applicare Hausdorff in 3D, costruiamo volumi 3D, combinando ogni segmentazione 2D con una profondità di slice come coordinata Z, per poi calcolare la distanza (ai fini della competizione la profondità di slice è settata a 0.1. Le posizioni previste/attese dei pixel sono normalizzate con la dimensione dell’immagine, in modo da ottenere un punteggio in 0-1.
Successivamente, vengono combinate le due metriche, con un peso di 0.4 per il coefficiente Dice e un peso di 0.6 per la distanza di Hausdorff.

### Submission File

Per ridurre la dimensione del file, la metrica proposta fa uso di encoding run-length sui valori dei pixel.
Verranno inviate coppie di valori contenenti una posizione iniziale e un valore di run (es. ‘1 3’ significa che inizia al pixel 1 e si estende per 3 pixel).
Va detto che, in fase di encoding, la maschera dovrebbe essere binaria, che implica una combinazione delle maschere di tutti gli oggetti dell’immagine in una singola maschera.
Le coppie vengono separate da spazi, per cui una lista ‘1 3 10 5’ implica che i pixel 1,2,3,10,11,12,13,14 saranno inclusi nella maschera.
La metrica controlla che le coppie siano ordinate, positive e che i valori decodificati dei pixel non siano duplicati.
Infine, va detto che i pixel sono numerati dall’alto in basso e da sinistra a destra (dunque 1 è (1,1), 2 è (2,1), ecc.)
Il file di invio dovrebbe contenere un header e avere la seguente formattazione:

```
id,class,predicted
1,large_bowel,1 1 5 1
1,small_bowel,1 1
1,stomach,1 1
2,large_bowel,1 5 2 17
etc.
```

# DATA

## Descrizione del Dataset
In questa competizione stiamo segmentando le cellule degli organi nelle immagini. Le annotazioni di addestramento sono fornite come maschere codificate in RLE e le immagini sono in formato PNG a 16 bit in scala di grigi.

Ogni caso in questa competizione è rappresentato da più serie di slice di scansione (ogni serie è identificata dal giorno in cui è stata effettuata la scansione). Alcuni casi sono divisi per tempo (i primi giorni sono in addestramento, i giorni successivi sono in test), mentre altri sono divisi per caso - l'intero caso è in addestramento o in test. L'obiettivo di questa competizione è quello di essere in grado di generalizzare sia a casi parzialmente che interamente non visti.

Si noti che, in questo caso, il test set è interamente inedito. Si tratta di circa 50 casi, con un numero variabile di giorni e di slice, come nel training set.

## Come funziona un test set completamente nascosto?

Il test set in questa competizione sarà disponibile soltanto quando verrà caricato il tuo codice. Il file sample_submission.csv fornito nel public set è un segnaposto vuoto che mostra il formato richiesto per l'invio; è necessario eseguire la modellazione, la cross-validation ecc., utilizzando il training set e scrivere il codice per elaborare un campione di invio non vuoto. Esso conterrà righe con le colonne id, classe e predicted.

Quando si invia il notebook, il codice viene eseguito sul test set non nascosto, che ha lo stesso formato di cartella (<case>/<case_day>/<scans>) dei dati di addestramento.

## Files

* train.csv - contenente ID e maschere per tutti gli oggetti di training
* sample_submission.csv - un esempio di file da inviare nel formato corretto
* train - una cartella di cartelle per caso/giornata, ognuna contenente delle porzioni di immagine per uno specifico caso nella specifica giornata
Bisogna tener nota che il nome del file include 4 numeri (es. 276_276_1.63_1.63.png), rispettivamente altezza/larghezza della porzione (pixel in interi) e altezza/larghezza della spaziatura (mm in numeri in virgola mobile). I primi due numeri definiscono la risoluzione della slide, mentre le altre due la dimensione fisica del pixel.

## Columns

* id - identificatore univoco per l’oggetto
* class - la classe prevista per l’oggetto
* EncodedPixels - pixel codificati in formato RLE per l’oggetto identificato

*Developed By **Vincenzo Fardella** & **Mario Tortorici** @ **Università degli Studi di Palermo***
