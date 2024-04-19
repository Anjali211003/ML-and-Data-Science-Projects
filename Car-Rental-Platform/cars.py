import datetime

class CarRental:
    
    def __init__(self,stock=0):
        """
    Our constructor class that instantiates car rental shop.
        """

        self.stock = stock

    def displaystock(self):
        """
        Displays the cars currently available for rent in the shop.
        """

        print("We have currently {} cars available to rent.".format(self.stock))
        return self.stock

    def rentCarOnHourlyBasis(self, n):
        """
        Rents a car on hourly basis to a customer.
        """
        if n <= 0:
            print("Number of cars should be positive!")
            return None
        
        elif n > self.stock:
            print("Sorry! We have currently {} cars available to rent.".format(self.stock))
            return None
        
        else:
            now_hourly = datetime.datetime.now()                      
            print("You have rented a {} car(s) on hourly basis today on : {}".format(n,now_hourly))
            print("You will be charged $5 for each hour per car.")
            print("We hope that you enjoy our service.")

            self.stock -= n
            return now_hourly      
     
    def rentCarOnDailyBasis(self, n):
        """
        Rents a car on daily basis to a customer.
        """
        if n <= 0:
            print("Number of cars should be positive!")
            return None

        elif n > self.stock:
            print("Sorry! We have currently {} cars available to rent.".format(self.stock))
            return None
    
        else:
            now_daily = datetime.datetime.now()                      
            print("You have rented {} car(s) on daily basis today on : {}".format(n, now_daily))
            print("You will be charged $20 for each day per car.")
            print("We hope that you enjoy our service.")

            self.stock -= n
            return now_daily
        
    def rentCarOnWeeklyBasis(self, n):
        """
        Rents a car on weekly basis to a customer.
        """
        if n <= 0:
            print("Number of cars should be positive!")
            return None

        elif n > self.stock:
            print("Sorry! We have currently {} cars available to rent.".format(self.stock))
            return None        
        
        else:
            now_weekly = datetime.datetime.now()
            print("You have rented {} car(s) on weekly basis today on : {}".format(n, now_weekly))
            print("You will be charged $60 for each week per car.")
            print("We hope that you enjoy our service.")
            self.stock -= n

            return now_weekly
    

    
    def returnCar(self, request):
        """
        1. Accept a rented car from a customer
        2. Replensihes the inventory
        3. Return a bill
        """
        rentalTime, rentalBasis, numOfCars = request
        bill = 0

        if rentalTime and rentalBasis and numOfCars:
            self.stock += numOfCars
            now = datetime.datetime.now()
            rentalPeriod = now - rentalTime
        
            # hourly bill calculation
            if rentalBasis == 1:
                bill = round(rentalPeriod.seconds / 3600) * 5 * numOfCars
                print(f"You have rented {numOfCars} cars on hourly basis and had returned on {now}")
                
            # daily bill calculation
            elif rentalBasis == 2:
                bill = round(rentalPeriod.days) * 20 * numOfCars
                print(f"You have rented {numOfCars} cars on daily basis and had returned on {now}")
                
            # weekly bill calculation
            elif rentalBasis == 3:
                bill = round(rentalPeriod.days / 7) * 60 * numOfCars
                print(f"You have rented {numOfCars} cars on weekly basis and had returned on {now}")


            print("Thanks for returning your car. Hope you enjoyed our service!")
            print("That would be ${}".format(bill))
            return bill
        else:
            print("Are you sure you rented a car with us?")
            return None


class Customer:

    def __init__(self):
        """
        Our constructor method which instantiates various customer objects.
        """
        
        self.cars = 0
        self.rentalBasis = 0
        self.rentalTime = 0
        self.bill = 0

    
    def requestCar(self):
        """
        Takes a request from the customer for the number of cars.
        """
                      
        cars = input("How many cars would you like to rent?")
        try:
            cars = int(cars)
        except ValueError:
            print("That's not a positive integer!")
            return -1

        if cars < 1:
            print("Invalid input. Number of cars should be greater than zero!")
            return -1
        else:
            self.cars = cars
        return self.cars
     
    def returnCar(self):
        """
        Allows customers to return their cars to the rental shop.
        """
        if self.rentalBasis and self.rentalTime and self.cars:
            return self.rentalTime, self.rentalBasis, self.cars  
        else:
            return 0,0,0
