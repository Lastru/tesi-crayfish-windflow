# tesi-crayfish-windflow
Repository della tesi su benchmarking di pipeline di streaming inference (Crayfish / WindFlow / Flink). Contiene codice, script di analisi e plotting, configurazioni degli esperimenti e risultati in formato CSV. Il README documenta l’ambiente e i comandi necessari per rigenerare tutte le figure a partire dai dati raccolti.

## Rigenerare i grafici dei risultati (JupyterLab)

I grafici dei risultati vengono generati tramite il notebook **`windflow_analysis.ipynb`**, eseguito in **JupyterLab** dalla cartella `results-analysis/`.  
Il notebook legge i file di output degli esperimenti (CSV) dalla cartella **`../results/`** e salva automaticamente le figure su file **dentro `results-analysis/`**.

---

### 1) Prerequisiti

- **Python 3.8+** (nel mio ambiente: Python 3.8.10)
- **pip**
- Pacchetti Python già installati (nel mio ambiente: `jupyterlab`, `numpy`, `pandas`, `matplotlib`, …)

### 2) Posizionamento dei risultati (input)

Il notebook si aspetta i file di risultati in:
`tesi-crayfish-windflow/results/`

Poiché JupyterLab viene lanciato da `results-analysis/`, i path nel notebook sono tipicamente relativi a:
`../results/...`

### 3) Avvio di JupyterLab

Entrare nella cartella `results-analysis/` e avviare JupyterLab:

```bash
cd tesi-crayfish-windflow/results-analysis
jupyter lab
```

### 4) Esecuzione del notebook e rigenerazione grafici

In JupyterLab aprire: **`windflow_analysis.ipynb`**

Per rigenerare tutti i grafici in modo automatico:
* **Kernel → Restart Kernel and Run All**

In alternativa, è possibile eseguire le celle manualmente in sequenza (come faccio normalmente io) con:
* **Shift + Invio**

### 5) Output

Le figure vengono salvate automaticamente su file dentro `results-analysis/`.

Se si desidera pulire gli output prima di rigenerarli, basta rimuovere manualmente i file immagine generati in precedenza (es. `.png`, `.pdf`) presenti in `results-analysis/`.

---

### Troubleshooting rapido

* **`ModuleNotFoundError`**: installare il pacchetto mancante con `python3 -m pip install <nome_pacchetto>`.
* **`FileNotFoundError` sui CSV**: verificare che i risultati siano in `tesi-crayfish-windflow/results/` e che dal notebook i path `../results/...` siano corretti.
* **Nessun file immagine generato**: controllare che le celle che producono i plot includano `savefig(...)` e che siano state eseguite.
