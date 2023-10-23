from recommendation.datasets.movielens import movieLensDataPreProcessing
from recommendation.datasets.beauty import beautyDataPreProcessing
from recommendation.algorithms.Ceaser.ceaserAlgo import callCeaser
from recommendation.algorithms.SasRec.main import SasRec
from recommendation.algorithms.SRGNN.main import main

class RecommendationAlgorithms:
    def __init__(self):
        self.getInputFromUser()

    def getInputFromUser(self):
        algoChoice = 0
        datasetChoice = 0
        algoDetails = {}
        self.datasetDetails = ''
        while int(algoChoice) <= 0 or int(algoChoice) >= 5:
            print("Please choose a number coresponding to the algorithm that you want to run")
            print("1. SASRec\n2. SR-GNN\n3. Ceaser\n4. Bert4Rec\n")
            algoChoice = int(input("Enter your choice (1, 2, 3, or 4): "))
            algoDetails = self.getCorrespondingAlgorithm(algoChoice)
            if algoChoice <= 0 or algoChoice >= 5:
                print("Invalid option please select again")
            else:
                while int(datasetChoice) <= 0 or int(datasetChoice) >= 3:    
                    print("Please choose a number coresponding for the dataset")
                    print("1. Movie Lens\n2. Amazon Beauty\n")
                    datasetChoice = int(input("Enter your choice (1 or 2): "))
                    self.datasetDetails = self.getCorrespondingDataset(int(datasetChoice))
        print('You chose')
        print('Algorithm - ' , algoDetails.get('algo'))
        print('Dataset - ' , self.datasetDetails)
        if self.datasetDetails == 'MovieLens':
            movieLensDataPreProcessing.process_movie_lens_data('./recommendation/datasets/movielens/ml-1m.zip')
        else:
            beautyDataPreProcessing.process_amazon_beauty_data('./recommendation/datasets/beauty/beauty.json.gz')
        algoDetails.get('fn')(self.datasetDetails)


    def getCorrespondingAlgorithm(self, choice):
        options = {
            1: {'algo': 'sasrec', 'fn' : self.sasrec},
            2: {'algo': 'srgnn', 'fn' : self.srgnn},
            3: {'algo': 'ceaser', 'fn' : self.ceaser},
            4: {'algo': 'bert4rec', 'fn' : self.bert4rec},
        }
        return options.get(choice)

    def getCorrespondingDataset(self, choice):
        options = {
            1: 'MovieLens',
            2: 'AmazonBeauty'
        }
        return options.get(choice)

    def sasrec(self, datasetName):
        print("You chose sasrec")
        SasRec(
            dataset= 'movielens' if datasetName == 'MovieLens' else 'beauty',
            train_dir='default',
            maxlen=200,
            dropout_rate=0.2,
            device='cuda'
        )

    def srgnn(self, datasetName):
        print("You chose srgnn")
        print("You chose the dataset - " , self.datasetDetails)
        main(dataset = 'movielens' if datasetName == 'MovieLens' else 'beauty',)

    def ceaser(self):
        print("You chose ceaser")
        callCeaser()

    def bert4rec(self):
        print("You chose bert4rec")
        print("You chose the dataset - " , self.datasetDetails)

    def default_case(self):
        print("Invalid option")
        print("You chose the dataset - " , self.datasetDetails)


recommendationAlgorithms = RecommendationAlgorithms()
