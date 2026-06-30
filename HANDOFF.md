# Handoff — Sistema de Predicción e Inteligencia Deportiva Mundial 2026

## 1. Estado actual

El proyecto evolucionó a un predictor híbrido: MLP/LSTM/GRU siguen siendo el backbone exigido por la rúbrica y una capa estadística añade calibración bayesiana, distribución Dixon–Coles de marcadores e incertidumbre posterior para Monte Carlo.

- Rama: `main`.
- Último commit previo a estas mejoras: `e78c797` (`Implement World Cup 2026 prediction system`).
- Las mejoras híbridas y este archivo todavía están sin commit.
- Python 3.13, TensorFlow 2.21 y PyMC 6.0.1.
- Tests: `16 passed`.
- Dashboard: verificado mediante `streamlit.testing`; cero excepciones.
- Pipeline `--quick`: ejecutado de extremo a extremo y genera artefactos v2 utilizables.
- Auditoría NUTS real: `R-hat máximo = 1.0`, `ESS bulk mínimo = 1408`, 4 cadenas.
- Entrenamiento final de 50,000 iteraciones: **pendiente de ejecutar**.
- El notebook editable está actualizado; `Proyecto_Mundial_2026_Ejecutado.ipynb` conserva resultados de la versión anterior y debe regenerarse después del entrenamiento final.

## 2. Comandos de reproducción

Instalación y datos:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
python scripts/build_dataset.py
```

Entrenamiento final —50,000 pasos ADVI, 64 muestras posteriores y auditoría NUTS completa—:

```bash
python scripts/train_models.py
```

La forma explícita equivalente es:

```bash
python scripts/train_models.py --bayes-steps 50000
```

Prueba de integración rápida —2 épocas, 200 pasos ADVI y sin NUTS—:

```bash
python scripts/train_models.py --quick
```

Reconstrucción histórica sin usar partidos posteriores al corte:

```bash
python scripts/build_dataset.py --as-of-date YYYY-MM-DD
```

La credencial Kaggle se lee desde `KAGGLE_API_TOKEN` en `.env`. Datos, credenciales y artefactos continúan ignorados por Git; en una copia nueva deben regenerarse.

## 3. Arquitectura actual

- `src/mundial/data.py`: joins as-of, rolling temporal, H2H, plantillas, ventanas y corte `as_of_date`.
- `src/mundial/models.py`: MLP de cuatro capas y encoders LSTM/GRU con tres cabezas multitarea.
- `src/mundial/statistical.py`: temperature scaling bayesiano, Dixon–Coles jerárquico dinámico, ADVI y auditoría NUTS.
- `src/mundial/training.py`: tres folds expansivos, selección por log-loss, test bloqueado, gate de calibración y reentrenamiento de producción.
- `src/mundial/inference.py`: artefactos v2, H2H real, simetría A/B y matrices de marcador 13×13.
- `src/mundial/simulation.py`: muestreo directo de marcadores, muestra posterior común por torneo, prórroga, penaltis e intervalos Monte Carlo.
- `app.py`: predictor, grupos, bracket y campeones con opciones de 100 a 20,000 simulaciones e IC 95%.

Distribución final por partido:

```text
P(marcador)
  = P_DL_calibrada(resultado)
  × P_Dixon-Coles(marcador | resultado)
```

El DL conserva exactamente la masa 1-X-2; Dixon–Coles distribuye esa masa entre marcadores coherentes. El último bucket de cada eje representa 12 o más goles para no perder masa de probabilidad.

Flujo de artefactos:

```text
data/raw/*
  → data/processed/matches.parquet + sequences.npz
  → artifacts/selected_model.keras
  → artifacts/inference_bundle.joblib
  → artifacts/calibration_posterior.joblib
  → artifacts/dixon_coles_posterior.joblib
  → artifacts/artifact_manifest.json
  → Streamlit / simulador
```

El cargador exige artefactos versión 2. Si falta alguno o la versión no coincide, entra en modo demostración de forma explícita.

## 4. Datos y protocolo de evaluación

Dataset actual:

- 11,043 partidos y 294 selecciones.
- 24 features estáticas y dos ventanas `(10, 5)`.
- Train histórico: 4,942 partidos.
- Validación histórica: 2,324 partidos.
- Test bloqueado: 64 partidos del Mundial 2022.
- Estado/producción: 3,713 partidos posteriores.

La selección ya no mira el Mundial 2022. Se usan tres folds expansivos:

1. Validación 2018–2019, entrenamiento anterior a 2018.
2. Validación 2020–2021, entrenamiento anterior a 2020.
3. Validación 2022 hasta el 19-11, entrenamiento anterior a 2022.

El backbone se selecciona por menor log-loss promedio; Brier desempata. Después se evalúa una sola vez sobre Mundial 2022 y se reentrena una versión de producción con todos los partidos disponibles hasta el corte.

## 5. Métricas actuales —solo corrida rápida—

Estas cifras provienen de 2 épocas y 200 pasos ADVI. Sirven para validar integración, **no para presentar resultados finales**.

Backtesting temporal:

| Modelo | Log-loss ↓ | Brier ↓ | Macro F1 |
|---|---:|---:|---:|
| MLP + Adam | **0.9092** | **0.5359** | 0.4379 |
| GRU | 0.9198 | 0.5422 | **0.4502** |
| LSTM | 0.9266 | 0.5461 | 0.4423 |
| MLP + SGD | 0.9562 | 0.5652 | 0.4060 |

Mundial 2022, modelo servido por la corrida rápida:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.5625 |
| Macro F1 | 0.4201 |
| Log-loss | 0.9848 |
| Brier | 0.5790 |
| ECE | 0.0710 |
| MAE goles A híbrido | 1.0931 |
| MAE goles B híbrido | 0.8454 |

La calibración candidata mejoró apenas log-loss/Brier, pero empeoró ECE y macro F1; el gate la rechazó y el artefacto servido usa calibración identidad. Dixon–Coles tampoco converge con solo 200 pasos, como era esperable. Hay que reemplazar estas cifras con la corrida final.

## 6. Cambios importantes respecto al sistema anterior

- El estado exportado incluye el último partido completado; ya no queda retrasado un encuentro.
- H2H real se exporta y usa en inferencia; ya no se reemplaza por ceros.
- Los partidos neutrales se aumentan intercambiando A/B y la inferencia promedia ambas orientaciones. La simetría se verificó con error cero.
- Resultado y marcador ya no se sortean por separado ni se corrigen artificialmente.
- En cada torneo se elige una única muestra posterior para todos sus partidos, conservando incertidumbre paramétrica coherente.
- Las eliminatorias simulan 90 minutos, prórroga con tasas escaladas a un tercio y penaltis si persiste el empate.
- Los intervalos de campeón usan Wilson al 95% y representan error Monte Carlo; no son intervalos deportivos totales.
- El último desempate de grupo sigue siendo aleatorio y reproducible porque no hay datos de fair play.

## 7. Limitaciones y riesgos pendientes

1. **Falta la corrida final.** Los artefactos actuales son de `--quick`; no deben usarse en la entrega final.
2. **Empates como argmax.** El backbone rápido sigue sin elegir empates como clase máxima, aunque sí asigna masa probabilística y el simulador la utiliza.
3. **Tamaño del test.** Mundial 2022 contiene solo 64 partidos; diferencias pequeñas necesitan bootstrap o intervalos para interpretarse.
4. **Ranking reciente.** Debe comprobarse que las 48 selecciones tengan el snapshot FIFA más reciente disponible.
5. **Plantillas.** FC 24 sigue siendo proxy; no se usan convocatorias oficiales 2026.
6. **Rendimiento.** La incertidumbre posterior aumenta el tiempo del simulador. Se verificó funcionalidad con 2,000 corridas, pero falta un benchmark y posible vectorización antes de usar 10,000/20,000 de forma interactiva.
7. **Notebook ejecutado.** Debe ejecutarse otra vez después del entrenamiento final para que tablas, curvas y métricas coincidan con `artifacts/metrics.json`.
8. **Cobertura de selecciones.** Aún conviene añadir una prueba de integración que exija ranking, plantilla y secuencia propios para las 48 selecciones.

## 8. Próximos pasos exactos

1. Ejecutar `python scripts/train_models.py` y conservar la salida completa.
2. Revisar `artifacts/metrics.json`: gate de calibración, ELBO, métricas híbridas y auditoría NUTS.
3. Confirmar `R-hat < 1.01`, `ESS bulk > 400` y estabilidad del ELBO.
4. Ejecutar `pytest -q` y abrir las cuatro vistas del dashboard.
5. Regenerar `notebooks/Proyecto_Mundial_2026_Ejecutado.ipynb` desde el notebook editable.
6. Sustituir en este handoff la tabla rápida por las métricas finales.
7. Grabar el video y preparar la defensa: leakage temporal, BPTT, log-loss/Brier, temperature scaling, Dixon–Coles y diferencia entre incertidumbre posterior y Monte Carlo.

## 9. Comprobaciones rápidas

```bash
# Artefactos y métricas
python -m json.tool artifacts/artifact_manifest.json
python -m json.tool artifacts/metrics.json

# Tests
pytest -q

# Dashboard
streamlit run app.py

# GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si aparece “Modo demostración”, comprobar los cinco artefactos v2 enumerados en la sección 3, no solo el `.keras` y el bundle.
