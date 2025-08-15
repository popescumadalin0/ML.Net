using Microsoft.ML;
using ML.Net.Models;
using Spectre.Console;

var ctx = new MLContext();

// load data
var dataView = ctx.Data
    .LoadFromTextFile<ActivityData>("TrainingData/training.csv", hasHeader: true, separatorChar: ';');

// split data into testing set
var splitDataView = ctx.Data
    .TrainTestSplit(dataView, testFraction: 0.2);

// Build model
var pipeline = ctx.Transforms.Conversion.MapValueToKey("Label", nameof(ActivityData.Outcome))
    .Append(ctx.Transforms.Text.FeaturizeText("ParamFeatures", nameof(ActivityData.Parameters)))
    .Append(ctx.Transforms.Text.FeaturizeText("TypeFeatures", nameof(ActivityData.Type)))
    .Append(ctx.Transforms.Text.FeaturizeText("NameFeatures", nameof(ActivityData.Name)))
    .Append(ctx.Transforms.Text.FeaturizeText("DescriptionFeatures", nameof(ActivityData.Description)))

    .Append(ctx.Transforms.Concatenate("Features", "ParamFeatures", "TypeFeatures", "NameFeatures", "DescriptionFeatures"))

    .Append(ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy())
    .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

// Train model
ITransformer model = default!;

var rule = new Rule("Create and Train Model");
AnsiConsole
    .Live(rule)
    .Start(console =>
    {
        // training happens here
        model = pipeline.Fit(splitDataView.TrainSet);
        var predictions = model.Transform(splitDataView.TestSet);

        rule.Title = "🏁 Training Complete, Evaluating Accuracy.";
        console.Refresh();

        // evaluate the accuracy of our model
        var metrics = ctx.BinaryClassification.Evaluate(predictions);

        var table = new Table()
            .MinimalBorder()
            .Title("💯 Model Accuracy");
        table.AddColumns("Accuracy", "Auc", "F1Score");
        table.AddRow($"{metrics.Accuracy:P2}", $"{metrics.AreaUnderRocCurve:P2}", $"{metrics.F1Score:P2}");

        console.UpdateTarget(table);
        console.Refresh();
    });

while (true)
{
    var text = AnsiConsole.Ask<string>("What are your [green]activity properties[/]?");
    var engine = ctx.Model.CreatePredictionEngine<ActivityData, ActivityPrediction>(model);

    var input = new ActivityData { Parameters = text, Type = "CreateDocument", Name = "CreateDocument" };
    var result = engine.Predict(input);

    AnsiConsole.MarkupLine($"{text} - {result.PredictedOutcome} {result.Probability:P00}");
}