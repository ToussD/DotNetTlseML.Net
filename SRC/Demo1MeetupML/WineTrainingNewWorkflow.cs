using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;

namespace Demo1MeetupML
{
    public static class WineTrainingNewWorkflow
    {
        public static void Train()
        {

            var env = new TlcEnvironment(new SysRandom(0));

            var loss = new SquaredLoss();

            var reader = TextLoader.CreateReader(env, ctx => (Id: ctx.LoadText(0)
                                                            , Country: ctx.LoadText(1)
                                                            , Description: ctx.LoadText(2)
                                                            , Designation: ctx.LoadText(3)
                                                            , Points: ctx.LoadFloat(4)
                                                            , Price: ctx.LoadFloat(5)
                                                            , Province: ctx.LoadText(6)
                                                            , Region1: ctx.LoadText(7)
                                                            , Region2: ctx.LoadText(8)
                                                            , TasterName: ctx.LoadText(9)
                                                            , TasterTwitter: ctx.LoadText(10)
                                                            , Title: ctx.LoadText(11)
                                                            , Variety: ctx.LoadText(12)
                                                            , Winery: ctx.LoadText(13))
                , hasHeader: true, separator: ',');

            string trainDataPath = "train.csv";

            var estimator = reader.MakeNewEstimator()
                .Append(row => (
                    price: row.Price,
                    country: row.Country.FeaturizeText(),
                    designation: row.Designation.FeaturizeText(),
                    description: row.Description.FeaturizeText(),
                    tasterName: row.TasterName.FeaturizeText(),
                    Province: row.Province.FeaturizeText(),
                    Region1: row.Region1.FeaturizeText(),
                    Region2: row.Region2.FeaturizeText(),
                    TasterName: row.TasterName.FeaturizeText(),
                    TasterTwitter: row.TasterTwitter.FeaturizeText(),
                    Title: row.Title.FeaturizeText(),
                    Variety: row.Variety.FeaturizeText(),
                    Winery: row.Winery.FeaturizeText(),
                    points: row.Points
                    )
                )
                .Append(row => (
                    features: row.country.ConcatWith(row.designation).ConcatWith(row.tasterName).ConcatWith(row.points).ConcatWith(row.description).Normalize(),
                    row.price))
                .Append(row => (
                    row.price,
                    score: row.price.PredictSdcaRegression(row.features, loss: loss)))
                .Append(row => (
                    predictedPrice: row.score,
                    row.price));

            var data = reader.Read(new MultiFileSource(trainDataPath));

            var model = estimator.Fit(data);

 
            var scores = model.Transform(reader.Read(new MultiFileSource(@"test.csv")));
            var metrics = RegressionEvaluator.Evaluate(scores, r => r.price, r => r.predictedPrice);

            Console.WriteLine("RSquared is: " + metrics.RSquared);

           
            var predictor = model.AsDynamic.MakePredictionFunction<Wine, WinePricePrediction>(env);

            var prediction = predictor.Predict(new Wine()
            {
                Country = "US",
                Description = @"Dusty mineral, smoke and struck flint lend a savory tone to this lean light-bodied Riesling. Off dry in style, the palate offers delicately concentrated flavors of red apple and nectarine off set by tangerine acidity. Drink now through 2021.",
                Designation = "Red Oak Vineyard",
                Points = 87,
                TasterName = "Anna Lee C. Iijima"
                //20
            });

            
            var prediction2 = predictor.Predict(new Wine()
            {
                Country = "France",
                Description = @"Produced from cru vines at the base of Mount Brouilly, the wine has structure as well as ripe black-plum fruits. It is generous and its fruit is well balanced by acidity and solid tannins. The wines is ready to drink.",
                Designation = "Les Quartelets",
                Points = 87,
                TasterName = "Roger Voss"
                //23
            });

            var prediction3 = predictor.Predict(new Wine()
            {
                Country = "Italy",
                Description = @"A blend of Cabernet Sauvignon, Merlot, Cabernet Franc and Sangiovese, this pleasant red has aromas of dark-skinned fruit, toast and a whiff of espresso. The light-bodied, straightforward palate offers cherry, red currant and a hint of light spice alongside zesty acidity and polished tannins.",
                Designation = "Castiglioni",
                Points = 87,
                TasterName = "Kerin O’Keefe"
                //30
            });

            Console.ReadLine();
        }

        public class Wine
        {
            public string Id;

            public string Country;

            public string Description;

            public string Designation;

            public float Points;
          
            public float Price;

            public float Label;

            public string Province;

            public string Region1;

            public string Region2;

            public string TasterName;

            public string TasterTwitter;

            public string Title;

            public string Variety;

            public string Winery;
        }

        public class WinePricePrediction
        {
            public float Score;
        }


    }
}
