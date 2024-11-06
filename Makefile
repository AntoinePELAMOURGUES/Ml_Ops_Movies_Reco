# Makefile for cleaning Docker resources

.PHONY: clean-containers clean-images clean-volumes clean-all

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
