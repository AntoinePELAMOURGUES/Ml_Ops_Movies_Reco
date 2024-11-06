# Makefile for cleaning Docker resources

.PHONY: clean-containers clean-images clean-volumes clean-all start stop

# Remove all stopped containers
clean-containers:
	@echo "Removing all stopped containers..."
	@docker container prune -f

# Remove unused images
clean-images:
	@echo "Removing unused images..."
	@docker image prune -f

# Remove unused volumes
clean-volumes:
	@echo "Removing unused volumes..."
	@docker volume prune -f

# Clean all: combines all clean commands
clean-all: clean-containers clean-images clean-volumes
	@echo "All Docker resources cleaned up."

# Start the containers
# Start all containers
start:
	@echo "Starting containers in the current directory..."
	docker-compose up -d
	@echo "##########################"
	@echo "Started services in the current directory."
	@echo "airflow: http://localhost:8080"
	@echo "streamlit: http://127.0.0.1:8501"
	@echo "fastapi: http://127.0.0.1:6060"
	@echo "MlFlow: http://127.0.0.1:5000"

	@echo "Starting Supabase containers..."
	cd supabase/docker && docker compose up -d
	@echo "##########################"
	@echo "Supabase services started."
	@echo "supabase: http://localhost:8000"

# Stop all containers
stop:
	docker-compose down --remove-orphans
	cd supabase/docker && docker compose down --remove-orphans