# Intent Based Search System

This project is an **Intent based search system** designed to demonstrate a machine learning-powered search system for categorizing products. It includes a pipeline for data ingestion, model training, and serving predictions via a REST API. The project is containerized using Docker and supports monitoring with Prometheus.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Environment Variables](#environment-variables)
7. [Monitoring](#monitoring)
8. [Future Improvements](#future-improvements)

---

## Features

- **Data Pipeline**: Automated data ingestion using Apache Airflow.
- **Model Training**: A machine learning model is trained to categorize products based on their names.
- **Model Serving**: A REST API built with FastAPI serves predictions.
- **Monitoring**: Prometheus is integrated for monitoring API requests.
- **Containerized Deployment**: Docker Compose is used to orchestrate services.

---

## Architecture

The project consists of the following components:

1. **Data Pipeline**:

   - Ingests product data and uploads it to an S3 bucket.
   - Managed by Apache Airflow.

2. **Model Training**:

   - Trains a machine learning model using `scikit-learn`.
   - Logs the model and metrics to an MLflow server.

3. **Model Serving**:

   - A FastAPI-based REST API serves predictions.
   - Includes a fallback dummy model if no trained model is available.

4. **Monitoring**:

   - Prometheus tracks API request metrics.

5. **Containerized Services**:
   - PostgreSQL for Airflow's metadata database.
   - MLflow server for model tracking.
   - FastAPI for serving predictions.

---

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed.
- AWS credentials with access to an S3 bucket.
- Python 3.11 (if running locally).

### Steps

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd ecommerce-search-mvp
   ```

2. **Set Up Environment Variables**: Create a `.env` file in the root directory with the following variables:

   ```env
   AWS_ACCESS_KEY_ID=<your-aws-access-key>
   AWS_SECRET_ACCESS_KEY=<your-aws-secret-key>
   AWS_DEFAULT_REGION=<your-aws-region>
   POSTGRES_USER=airflow
   POSTGRES_PASSWORD=airflow
   ```

3. **Build and Start Services**: Navigate to the `docker` directory and run:

   ```bash
   docker-compose up --build
   ```

4. **Access Services**:
   - **Airflow Webserver**: [http://localhost:8080](http://localhost:8080)
   - **MLflow Server**: [http://localhost:5001](http://localhost:5001)
   - **FastAPI**: [http://localhost:8000](http://localhost:8000)

---

## Usage

### 1. Train the Model

Run the training script to train and log the model:

```bash
docker exec -it <container_name> python app/model/train.py
```

Replace `<container_name>` with the name of the API container.

### 2. Serve Predictions

Access the prediction API:

```bash
curl "http://localhost:8000/predict?query=red%20shoes"
```

### 3. Monitor API Requests

Prometheus metrics are available at [http://localhost:8001](http://localhost:8001).

---

## Project Structure

```plaintext
ecommerce-search-mvp/
├── app/
│   ├── dags/                # Airflow DAGs for data pipeline
│   ├── model/               # Model training and serving scripts
│   ├── data.csv             # Sample product data
│   ├── requirements.txt     # Python dependencies
├── docker/
│   ├── docker-compose.yml   # Docker Compose configuration
│   ├── api.Dockerfile       # Dockerfile for FastAPI
│   ├── airflow.Dockerfile   # Dockerfile for Airflow
├── monitoring/
│   ├── prometheus.yml       # Prometheus configuration
├── .env                     # Environment variables (not committed)
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
```

---

## Environment Variables

The following environment variables are required:

| Variable                | Description                      |
| ----------------------- | -------------------------------- |
| `AWS_ACCESS_KEY_ID`     | AWS access key for S3.           |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3.           |
| `AWS_DEFAULT_REGION`    | AWS region for S3.               |
| `POSTGRES_USER`         | PostgreSQL username for Airflow. |
| `POSTGRES_PASSWORD`     | PostgreSQL password for Airflow. |

---

## Monitoring

Prometheus is used to monitor API requests. Metrics are exposed at `/metrics` on port `8001`.

---

## Future Improvements

- Add support for more advanced machine learning models.
- Implement a frontend for user interaction.
- Enhance security by using secrets management for sensitive credentials.
- Add CI/CD pipelines for automated testing and deployment.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributors

- **Your Name** - Fahadkabir
