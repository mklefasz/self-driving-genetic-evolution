 
import math
import pygame
import os
import numpy as np
import random
 
class Car:
    def __init__(self, x, y):

      
        
        self.original_image = pygame.Surface((20, 50), pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, "red", (0, 0, 20, 50))
        pygame.draw.rect(self.original_image, "blue", (0, 0, 20, 10))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))

        self.position = pygame.math.Vector2(x, y) 
        self.velocity = 0
        self.angle = 0
        
        self.max_speed = 300
        self.acceleration = 10
        self.friction = 5 
        self.rotation_speed = 3
        
        self.sensor_angles = [0, 30, -30, 90, -90]
        self.sensor_length = 150
        self.sensor_data = [] 

        self.alive = True
        self.distance_traveled = 0
        self.steps =0


    def get_input(self):
        """Checks keys and updates speed/angle variables."""
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            self.angle += self.rotation_speed
        if keys[pygame.K_d]:
            self.angle -= self.rotation_speed
        
        if keys[pygame.K_w]:
            self.velocity = min(self.velocity + self.acceleration, self.max_speed)
        elif keys[pygame.K_s]:
            self.velocity = max(self.velocity - self.acceleration, -self.max_speed)
        else:
            if self.velocity > 0:
                self.velocity -= self.friction
                if self.velocity < 0: self.velocity = 0
            elif self.velocity < 0:
                self.velocity += self.friction
                if self.velocity > 0: self.velocity = 0

    def move(self,dt):
        """Updates the position based on the current speed and angle."""
        rad = math.radians(self.angle)
        
        dx = math.sin(rad) * self.velocity * dt
        dy = math.cos(rad) * self.velocity * dt

        self.position.x += dx
        self.position.y += dy

        self.rect.center = int(self.position.x), int(self.position.y)

    def drive(self, output_layers):
       
        
        turn_command = output_layers[0] 
        if turn_command > 0.5:
            self.angle -= self.rotation_speed 
        if turn_command < -0.5:
            self.angle += self.rotation_speed 

        throttle_command = output_layers[1]
        if throttle_command > 0:
            self.velocity = min(self.velocity + self.acceleration, self.max_speed)
        else:
            if self.velocity > 0:
                self.velocity -= self.friction
        
        

        if self.velocity > 0:
             self.distance_traveled += 1

    def check_collision(self,surface):

        center_x = int(self.position.x)
        center_y = int(self.position.y)

        
        if not (0 <= center_x < surface.get_width() and 0 <= center_y < surface.get_height()):
            self.alive = False
            return 
            
        try:
            pixel_color = surface.get_at((center_x, center_y))
        except IndexError:
             
            self.alive = False
            return

        
        if pixel_color[0] < 10 and pixel_color[1] < 10 and pixel_color[2] < 10:
            self.alive = False


    def check_sensors(self, surface):
        

        self.sensor_data = [] 
        
        for angle_offset in self.sensor_angles:
            detect_angle = math.radians(self.angle + angle_offset)
            
            start_pos = self.position
            
            # End of the ray (max distance)
            end_x = start_pos.x + math.sin(detect_angle) * self.sensor_length
            end_y = start_pos.y + math.cos(detect_angle) * self.sensor_length
            end_pos = (end_x, end_y)

            contact_point = end_pos 
            distance = self.sensor_length
             
            for i in range(1, int(self.sensor_length)):
                check_x = int(start_pos.x + math.sin(detect_angle) * i)
                check_y = int(start_pos.y + math.cos(detect_angle) * i)

                
                if not (0 <= check_x < surface.get_width() and 0 <= check_y < surface.get_height()):
                    break 
                
                pixel_color = surface.get_at((check_x, check_y))
                
                if pixel_color[0] < 10 and pixel_color[1] < 10 and pixel_color[2] < 10:
                    contact_point = (check_x, check_y)
                    distance = i
                    pygame.draw.circle(screen, "red", contact_point, 5) # Draw red dot at hit
                    break 
            
            pygame.draw.line(screen, "green", start_pos, contact_point)
            self.sensor_data.append(distance)

        

    def draw(self, surface):
        """Rotates the car image and puts it on screen."""
        rotated_image = pygame.transform.rotate(self.original_image, self.angle)
        new_rect = rotated_image.get_rect(center=self.rect.center)
        surface.blit(rotated_image, new_rect)

     


 

class Genomem:
    def __init__(self, mutation_strength,total_weight_count):
        self.weights = np.array([random.randint(-1,1) for rand in range(total_weight_count)])
        self.mutation_strength = mutation_strength
        self.fitness = None


    def mutate_with_rate(self, mutation_rate=0.1, mutation_strength=0.1):
  
        for i in range(len(self.weights)):
        
            if random.random() < mutation_rate:
             
                self.weights[i] += random.gauss(0, mutation_strength)

    def get_weights(self):
        return self.weights
    
class NeuralNetwork:
    def __init__(self,genome,input_size=5, hidden_size=8, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


        self.genome = genome
        self.W1_size = input_size * hidden_size
        self.B1_size = hidden_size
        self.W2_size = hidden_size * output_size
        self.B2_size = output_size
        
        self.total_weights = self.W1_size + self.B1_size + self.W2_size + self.B2_size

    def _activation_fn(self, Z):
        return np.tanh(Z)
    
    def _unpack_weights(self):

        weights = self.genome.get_weights()

        W1 = weights[:self.W1_size].reshape(self.input_size,self.hidden_size)

        B1 = weights[self.W1_size : self.W1_size + self.B1_size]

        W2 = weights[self.W1_size + self.B1_size : self.W1_size + self.B1_size+self.W2_size].reshape(self.hidden_size,self.output_size)
        B2 = weights[self.W1_size + self.B1_size+self.W2_size : ] 

        return W1,B1,W2,B2


    def forward(self,input):

        W1, B1, W2, B2 = self._unpack_weights()
        
        np_input = np.atleast_2d(input)

        Z1 = np_input @ W1 + B1
     
        A1 = self._activation_fn(Z1) 
     
    
     
        Z2 = A1 @ W2 + B2
    
     
        A2 = self._activation_fn(Z2)
    
        return A2

 
 

def crossover(parent1_weights, parent2_weights):
    """Combines two weight arrays at a random point."""
    size = len(parent1_weights)
    crossover_point = random.randint(int(size * 0.1), int(size * 0.9)) 
    
    
    child_weights = np.concatenate([
        parent1_weights[:crossover_point],
        parent2_weights[crossover_point:]
    ])
    
    return child_weights
 
def get_fitness(car):
     
    distance_reward = car.distance_traveled 
     
    if car.steps > 0:
        average_velocity = distance_reward / car.steps
         
        if average_velocity < 0.1:
             time_penalty = car.steps * 0.5  
        else:
             time_penalty = 0
    else:
         time_penalty = 0
 
    final_fitness = distance_reward - time_penalty
    
    return final_fitness
 
def next_generation(old_population):
    """Handles selection, crossover, and mutation."""
     
    old_population.sort(key=lambda g: g.fitness, reverse=True)
     
    best_fitness = old_population[0].fitness
    avg_fitness = np.mean([g.fitness for g in old_population])
    

   
    num_elite = int(0.1 * POPULATION_SIZE)
    new_population = old_population[:num_elite]
    
    
    while len(new_population) < POPULATION_SIZE:

        parent1 = random.choice(old_population[:POPULATION_SIZE // 2]) 
        parent2 = random.choice(old_population[:POPULATION_SIZE // 2])

        
        child_weights = crossover(parent1.get_weights(), parent2.get_weights())
        
        
        child_genome = Genomem(MUTATION_STRENGTH, TOTAL_WEIGHT_COUNT)
        child_genome.weights = child_weights
        
        
        child_genome.mutate_with_rate(mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH)
        
        new_population.append(child_genome)

    return new_population, best_fitness, avg_fitness

 
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

 
image_path = r"track.png"
raw_image = pygame.image.load(image_path)
background = pygame.transform.scale(raw_image, (1280, 720))
 
player = Car(200, 360)
 
 
POPULATION_SIZE = 50
MUTATION_STRENGTH = 0.5 
MUTATION_RATE = 0.05
INPUT_SIZE = 5
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2
START_X, START_Y = 200, 360

TOTAL_WEIGHT_COUNT = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + \
                     (HIDDEN_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE


current_genomes = [Genomem(MUTATION_STRENGTH, TOTAL_WEIGHT_COUNT) for _ in range(POPULATION_SIZE)]
current_time_limit =10
cars = []
for genome in current_genomes:
    new_car = Car(START_X, START_Y)
     
    new_car.genome = genome
    new_car.network = NeuralNetwork(genome, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
    cars.append(new_car)

generation_num = 1
best_fitness_history = 0

generation_timer = 0.0 
MAX_GENERATION_TIME = 5

SIMULATION_SPEED_MULTIPLIER = 2
running = True
while running:
     
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
 
    dt = clock.tick(60) / 1000
 
    for _ in range(SIMULATION_SPEED_MULTIPLIER): 
        
        generation_timer += dt 
        
        active_cars = [car for car in cars if car.alive]
         
        if not active_cars or generation_timer >= MAX_GENERATION_TIME:
             
            for car in cars:
                if car.genome.fitness is None: 
                    car.genome.fitness = get_fitness(car)
             
            current_genomes, best_fit, avg_fit = next_generation(current_genomes)
             
            best_fitness_history = max(best_fitness_history, best_fit)
             
            cars = []
            for genome in current_genomes:
                new_car = Car(START_X, START_Y)
                new_car.genome = genome
                new_car.network = NeuralNetwork(genome, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
                cars.append(new_car)
                
            generation_num += 1
            generation_timer = 0.0  

            if generation_num % 5 == 0:
                current_time_limit = min(
                    current_time_limit + 5, 
                    60
                )
                time_increase_message = " (TIME INCREASED)"
            else:
                time_increase_message = ""

            print(f"Generation {generation_num} | Best Fit: {best_fit:.2f} | Avg Fit: {avg_fit:.2f}")

             
            break 
 
        for car in active_cars:
             
            car.check_sensors(background)
             
            normalized_inputs = [1.0 - (d / car.sensor_length) for d in car.sensor_data]
             
            nn_output = car.network.forward(normalized_inputs)[0].tolist() 
            
            car.drive(nn_output) 
            car.move(dt)
            car.check_collision(background)
            car.steps += 1 
         
    screen.blit(background, (0, 0))
    for car in active_cars:
        car.draw(screen)
 
    text_gen = font.render(f"Generation: {generation_num}", True, "white")
    text_alive = font.render(f"Alive: {len(active_cars)}/{POPULATION_SIZE} ({generation_timer:.1f}s)", True, "white")
    text_best = font.render(f"Best Ever Fit: {best_fitness_history:.2f}", True, "white")

    if generation_num%50 is 0:
        MAX_GENERATION_TIME+=5
        
    
    screen.blit(text_gen, (50, 50))
    screen.blit(text_alive, (50, 80))
    screen.blit(text_best, (50, 110))
    
    pygame.display.flip()
 
pygame.quit()