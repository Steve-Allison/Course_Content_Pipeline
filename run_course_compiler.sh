#!/bin/bash

# Set these as needed:
PROJECT_DIR="/Users/steveallison/AI_Projects+Code/Course_content_compiler"
CONDA_ENV="course_pipeline"

echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "Project folder not found!"; exit 1; }

echo "Running main.py pipeline..."
python main.py

echo ""
echo "Pipeline finished. Check the /output folder for results and logs."
read -p "Press [Enter] to close this window."
