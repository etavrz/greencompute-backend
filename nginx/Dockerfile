# Use the official Nginx base image
FROM nginx:alpine

# Copy the Nginx configuration file to the container
COPY index.html /etc/nginx/html/index.html
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80 to the host
EXPOSE 80
