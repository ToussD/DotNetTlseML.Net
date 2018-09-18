using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Demo1MeetupML
{
    public class SaveAndLoadSample
    {

        public static void Train()
        {
            var SentimentDataPath = "wikipedia-detox-250-line-data.tsv";
            var dataPath = GetDataPath(SentimentDataPath);
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<SentimentData>(trimWhitespace: true));
            pipeline.Add(new TextFeaturizer("Features", "SentimentText")
            {
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizerTransformCaseNormalizationMode.Lower,
                OutputTokens = true,
                StopWordsRemover = new PredefinedStopWordsRemover(),
                VectorNormalizer = TextTransformTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }
            });

            pipeline.Add(new LightGbmBinaryClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            var model = pipeline.Train<SentimentData, SentimentPrediction>();

            var modelName = "trainSaveAndPredictdModel.zip";
            DeleteOutputPath(modelName);
            model.WriteAsync(modelName);

            var loadedModel = PredictionModel.ReadAsync<SentimentData, SentimentPrediction>(modelName).Result;
            var singlePrediction = loadedModel.Predict(new SentimentData() { SentimentText = "Love is beautiful" });

            Console.WriteLine($"{singlePrediction.PredictedLabel} {singlePrediction.Score}");
            Console.ReadLine();
        }

        private static void DeleteOutputPath(string modelName)
        {
            File.Delete(modelName);
        }

        public static string GetDataPath(string name)
        {
            var currentAssemblyLocation = new FileInfo(typeof(Program).Assembly.Location);
            var _rootDir = currentAssemblyLocation.Directory.Parent.Parent.Parent.Parent.FullName;
            var _outDir = Path.Combine(currentAssemblyLocation.Directory.FullName, "TestOutput");
            Directory.CreateDirectory(_outDir);
            var _dataRoot = Path.Combine(_rootDir);

            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(_dataRoot, "Demo1MeetupML", name));
        }

        public class SentimentData
        {
            [Column("0", name: "Label")]
            public bool Sentiment;
            [Column("1")]
            public string SentimentText;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedLabel;

            [ColumnName("Score")]
            public float Score;
        }
    }
}