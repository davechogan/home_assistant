   # Use an official Node.js runtime as a parent image
   FROM node:14

   # Set the working directory in the container
   WORKDIR /usr/src/app

   # Copy the current directory contents into the container at /usr/src/app
   COPY . .

   # Install any needed packages specified in package.json
   RUN npm install

   # Make port 3000 available to the world outside this container
   EXPOSE 3000

   # Run the application
   CMD ["npm", "start"]
