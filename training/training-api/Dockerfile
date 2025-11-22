FROM node:20-slim

WORKDIR /app

# Copy the whole repo so local backend and manifests can access training/common.
COPY . /app

WORKDIR /app/training/training-api

RUN npm install --production

EXPOSE 4000

CMD ["npm", "start"]
