class RecommendationAlgorithms:
    def __init__(self):
        self.getInputFromUser()

    def getInputFromUser(self):
        choice = 0
        while int(choice) <= 0 or int(choice) >= 5:
            print("Please choose a number coresponding to the algorithm that you want to run")
            print("1. SASRec\n2. SR-GNN\n3. Ceaser\n4. Bert4Rec\n")
            choice = input("Enter your choice (1, 2, 3, or 4): ")
            self.getCorrespondingAlgorithm(int(choice))

    def getCorrespondingAlgorithm(self, choice):
        options = {
            1: self.sasrec,
            2: self.srgnn,
            3: self.ceaser,
            4: self.bert4rec
        }
        options.get(choice, self.default_case)()

    def sasrec(self):
        print("You chose sasrec\n")

    def srgnn(self):
        print("You chose srgnn\n")

    def ceaser(self):
        print("You chose ceaser\n")

    def bert4rec(self):
        print("You chose bert4rec\n")

    def default_case(self):
        print("Invalid option")


recommendationAlgorithms = RecommendationAlgorithms()
