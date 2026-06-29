#!/usr/bin/env python3
"""Entrena MLP Adam/SGD, LSTM y GRU y exporta el ganador."""

import argparse

from mundial.training import train_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Usa dos epocas para una prueba de integracion")
    args = parser.parse_args()
    epochs_mlp, epochs_recurrent = (2, 2) if args.quick else (60, 40)
    result = train_all(max_epochs_mlp=epochs_mlp, max_epochs_recurrent=epochs_recurrent)
    print(f"Modelo seleccionado: {result['selected_model']}")

