#!/usr/bin/env python3
"""Entrena MLP Adam/SGD, LSTM y GRU y exporta el ganador."""

import argparse

from mundial.training import train_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Usa dos epocas para una prueba de integracion")
    parser.add_argument("--skip-nuts", action="store_true", help="Omite la auditoria NUTS de cuatro cadenas")
    parser.add_argument("--bayes-steps", type=int, default=50_000, help="Iteraciones ADVI para calibracion y Dixon-Coles")
    args = parser.parse_args()
    epochs_mlp, epochs_recurrent = (2, 2) if args.quick else (60, 40)
    bayes_steps = min(args.bayes_steps, 200) if args.quick else args.bayes_steps
    result = train_all(
        max_epochs_mlp=epochs_mlp,
        max_epochs_recurrent=epochs_recurrent,
        bayes_steps=bayes_steps,
        run_nuts_audit=not (args.skip_nuts or args.quick),
    )
    print(f"Modelo seleccionado: {result['selected_model']}")
