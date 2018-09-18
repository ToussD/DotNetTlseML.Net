using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Demo1MeetupML
{
    public static class WineTraining
    {
        public static void Train()
        {
            var pipeline = new LearningPipeline();

            string dataPath = "train.csv";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<Wine>(separator: ',', useHeader: true));
            pipeline.Add(new RowSkipAndTakeFilter() { Take = 600 });
            pipeline.Add(new TextFeaturizer("TextFeatures", "Title", "Designation"));
            pipeline.Add(new TextFeaturizer("DescriptionFeatures", "Description") { OutputTokens = true });
            pipeline.Add(new WordEmbeddings(("DescriptionFeatures_TransformedText", "DescriptionFeatures_WordEmbeddings")) { ModelKind = WordEmbeddingsTransformPretrainedModelKind.GloVe100D });
            pipeline.Add(new CategoricalOneHotVectorizer("Country", "Province", "Region1", "Region2", "TasterName", "TasterTwitter", "Variety", "Winery"));
            pipeline.Add(new ColumnConcatenator("Features", "Points", "Country", "Province", "Region1", "Region2", "TasterName", "TasterTwitter", "Variety", "Winery", "TextFeatures", "DescriptionFeatures_WordEmbeddings"));
            pipeline.Add(new MeanVarianceNormalizer("Features"));
            pipeline.Add(new LightGbmRegressor());

            var model = pipeline.Train<Wine, WinePricePrediction>();


            string dataPathtest = "test.csv";
            TextLoader test = new TextLoader(dataPathtest).CreateFrom<Wine>(separator: ',', useHeader: true);
            RegressionEvaluator evaluator = new RegressionEvaluator();

            Console.WriteLine("=============== Evaluating model ===============");

            RegressionMetrics metrics = evaluator.Evaluate(model, test);

            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
            Console.WriteLine("=============== End evaluating ===============");

            var prediction = model.Predict(new Wine()
            {
                Country = "US",
                Description = @"Dusty mineral, smoke and struck flint lend a savory tone to this lean light-bodied Riesling. Off dry in style, the palate offers delicately concentrated flavors of red apple and nectarine off set by tangerine acidity. Drink now through 2021.",
                Designation = "Red Oak Vineyard",
                Points = 87,
                TasterName = "Anna Lee C. Iijima"
                //20
            });

            
            var prediction2 = model.Predict(new Wine()
            {
                Country = "France",
                Description = @"Produced from cru vines at the base of Mount Brouilly, the wine has structure as well as ripe black-plum fruits. It is generous and its fruit is well balanced by acidity and solid tannins. The wines is ready to drink.",
                Designation = "Les Quartelets",
                Points = 87,
                TasterName = "Roger Voss"
                //23
            });

            var prediction3 = model.Predict(new Wine()
            {
                Country = "Italy",
                Description = @"A blend of Cabernet Sauvignon, Merlot, Cabernet Franc and Sangiovese, this pleasant red has aromas of dark-skinned fruit, toast and a whiff of espresso. The light-bodied, straightforward palate offers cherry, red currant and a hint of light spice alongside zesty acidity and polished tannins.",
                Designation = "Castiglioni",
                Points = 87,
                TasterName = "Kerin O’Keefe"
                //30
            });

            Console.WriteLine($"The prediction is {prediction.Label} {prediction.Score}");

            Console.ReadLine();

        }

        public class Wine
        {
            [Column("0")]
            public string Id;

            [Column("1")]
            public string Country;

            [Column("2")]
            public string Description;

            [Column("3")]
            public string Designation;

            [Column("4")]
            public float Points;

            [Column("5")]
            [ColumnName("Label")]
            public float Label;

            [Column("6")]
            public string Province;

            [Column("7")]
            public string Region1;

            [Column("8")]
            public string Region2;

            [Column("9")]
            public string TasterName;

            [Column("10")]
            public string TasterTwitter;

            [Column("11")]
            public string Title;

            [Column("12")]
            public string Variety;

            [Column("12")]
            public string Winery;
        }

        public class WinePricePrediction
        {
            [ColumnName("Score")]
            public float Score;

            [ColumnName("Label")]
            public float Label;
        }

    }
}
