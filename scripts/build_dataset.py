#!/usr/bin/env python3
"""Construye el dataset tabular y las ventanas de secuencia."""

from mundial.data import save_processed_dataset


if __name__ == "__main__":
    table, sequences = save_processed_dataset()
    print(f"Dataset tabular: {table}")
    print(f"Ventanas temporales: {sequences}")

