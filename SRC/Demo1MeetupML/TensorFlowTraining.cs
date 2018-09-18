using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Demo1MeetupML
{
    public static class TensorFlowTraining
    {
        public static void Train()
        {
            var imageHeight = 32;
            var imageWidth = 32;
            var model_location = "cifar_model/frozen_model2.pb";
            var dataFile = GetDataPath("Demo1MeetupML/images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(dataFile).CreateFrom<CifarData>());

            pipeline.Add(new ImageLoader(("ImagePath", "ImageReal"))
            {
                ImageFolder = imageFolder
            });

            pipeline.Add(new ImageResizer(("ImageReal", "ImageCropped"))
            {
                ImageHeight = imageHeight,
                ImageWidth = imageWidth,
                Resizing = ImageResizerTransformResizingKind.IsoCrop
            });

            pipeline.Add(new ImagePixelExtractor(("ImageCropped", "Input"))
            {
                UseAlpha = false,
                InterleaveArgb = true
            });

            pipeline.Add(new TensorFlowScorer()
            {
                ModelFile = model_location,
                InputColumns = new[] { "Input" },
                OutputColumns = new[] { "Output" },
            });


            pipeline.Add(new ColumnConcatenator("Features", "Input"));
            pipeline.Add(new TextToKeyConverter("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            PredictionModel<CifarData, CifarPrediction> model = pipeline.Train<CifarData, CifarPrediction>();

            Console.WriteLine("#########BANANA#######");
            predict("Demo1MeetupML/images/banana4.jpg", model);
            Console.WriteLine("#########HOTDOG#######");
            predict("Demo1MeetupML/images/hotdog4.jpg", model);
            predict("Demo1MeetupML/images/hotdog5.jpg", model);
            predict("Demo1MeetupML/images/hotdog6.jpg", model);
            predict("Demo1MeetupML/images/hotdog7.jpg", model);
            Console.WriteLine("#########TOMATO#######");
            predict("Demo1MeetupML/images/tomato4.jpg", model);
            predict("Demo1MeetupML/images/tomato5.jpg", model);
            predict("Demo1MeetupML/images/tomato6.jpg", model);
            predict("Demo1MeetupML/images/tomato7.jpg", model);

            Console.ReadLine();
        }

        public static void predict(string path, PredictionModel<CifarData, CifarPrediction> model)
        {
            CifarPrediction prediction = model.Predict(new CifarData()
            {
                ImagePath = GetDataPath(path)
            });

            model.TryGetScoreLabelNames(out var scoreLabels);

            Console.WriteLine($"{path}");
            Console.WriteLine($"{prediction.PredictedLabels[0]};{prediction.PredictedLabels[1]};{prediction.PredictedLabels[2]}");

            if (prediction.PredictedLabels[0] == prediction.PredictedLabels.Max())
            {
                Console.WriteLine(scoreLabels[0]);
            }
            else if (prediction.PredictedLabels[1] == prediction.PredictedLabels.Max())
            {
                Console.WriteLine(scoreLabels[1]);
            }
            else if (prediction.PredictedLabels[2] == prediction.PredictedLabels.Max())
            {
                Console.WriteLine(scoreLabels[2]);
            }

        }

        public class CifarData
        {
            [Column("0")] public string ImagePath;

            [Column("1")] public string Label;
        }

        public class CifarPrediction
        {
            [ColumnName("Score")] public float[] PredictedLabels;
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
            return Path.GetFullPath(Path.Combine(_dataRoot, name));
        }

    }
}
