# POWER OUTPUT PREDICTION
power output prediction using sklearn, FastAPI and Streamlit App

## Table of Contents
- [Description](#description)
- [Requirement](#requirement)
- [Getting started](#getting-started)
-[1. Train and save the model](#1-train-and-save-the-model)
- [2. Deploy FastAPI](#2-deploy-fastapi)
- [3. Run Streamlit](#run-streamlit)
- [Usage](#usage)
- [Endpoint](#endpoint)
- [Example Input and Output](#example-input-and-output)
- [File Structure](#file-structure)
- [License](#license)

## DESCRIPTION
This project provide an API and a streamlit application for predicting power output (PE) based on environmental factors. The model uses Linear Regression from scikit-Learn, trained on features including:

- Ambient Temperature (AT)
- Exhaust Vacuum(V)
- Ambient Pressure (AP)
- Relative Humidity (RH)

The API is deployed using FastAPI, and a stream app provides an interactive interface for users to input values and get predictions.