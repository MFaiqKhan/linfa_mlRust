use csv::ReaderBuilder;
use linfa::prelude::*; // import all items from the prelude module
use linfa_logistic::LogisticRegression;
use ndarray::{prelude::*, OwnedRepr};
use ndarray_csv::Array2Reader; // Array2Reader allow us to read a 2D array from a CSV file.
use plotlib::{
    grid::Grid,
    page::Page,
    repr::Plot,
    style::{PointMarker, PointStyle},
    view::{ContinuousView, View},
};

fn main() {
    // the below function reads teh data from the specified file , parses it and returns a Dataset object.
    let mut train = load_data("data/train.csv");
    let mut test = load_data("data/test.csv");

    let features = train.nfeatures(); //return number of features in the dataset
    let targets = train.ntargets(); //return the number of targets in the dataset


    let train_records = train.records().to_owned(); // returns the records of the dataset
    let test_records = test.records().to_owned();

    /* let train_targets = train.targets().slice(s![.., -1, 0]).into_shape([train.nsamples()]).to_owned();
    let test_targets = test.targets().slice(s![.., -1, 0]).into_shape([test.nsamples()]).to_owned();

    let train_targets = train.targets().slice(s![.., -1]).into_shape([train.nsamples()]).to_owned();
    let test_targets = test.targets().slice(s![.., -1]).into_shape([test.nsamples()]).to_owned(); */

   /*  let train_targets = train.targets().slice(s![.., -1, 0]).into_shape([train.nsamples()]).to_owned().expect("error message");
    let test_targets = test.targets().slice(s![.., -1, 0]).into_shape([test.nsamples()]).to_owned().expect("error message");    
 */
    //change the above ViewRpr to OwnedRepr

/*     let train_targets = train.targets().slice(s![.., -1, 0]).to_owned();
    let test_targets = test.targets().slice(s![.., -1, 0]).to_owned(); */

    /* let train_targets = train.targets().slice(s![.., -1, 0]).into_shape([train.nsamples()]).to_owned();
    let test_targets = test.targets().slice(s![.., -1, 0]).into_shape([test.nsamples()]).to_owned();
 */

 /* let train_targets = train.targets().slice(s![.., 0]).into_shape([train.nsamples()]).to_owned();
 let test_targets = test.targets().slice(s![.., 0]).into_shape([test.nsamples()]).to_owned(); */
 

 let train_targets = train.targets().slice(s![.., 0]).into_shape([train.nsamples()]).to_owned();


    println!(
        "training with {} samples, testing with {} samples, {} features and {} targets",
        train.nsamples(),
        test.nsamples(),
        features,
        targets
    );

    /* nsamples() and nfeatures() are both methods provided by the linfa library for working with datasets.
    They are used to retrieve different pieces of information about the dataset:
    nsamples() returns the number of samples in the dataset. A sample is a single row of data in the dataset,
    usually representing an instance of a phenomenon being studied or a single measurement.
    nfeatures() returns the number of features (or columns) in the dataset. A feature is a single variable or
    attribute of the dataset. In other words, it is a single element of information that describes a sample.
    For example, if you have a dataset of housing prices that includes the number of bedrooms,
    the number of bathrooms, and the square footage of each house as features, and you have 100 houses in your dataset,
    then nsamples() would return 100 and nfeatures() would return 3.So nfeatures are the number of
    columns/features in the dataset and nsamples represents the number of rows in the dataset */

    println!("Plotting the training data");
    plot_data(&train);

    // training the model and testing it
    println!("Training and Testing Model...");

    // if the data is 0.01  then it will be classified as "accepted" otherwise it will be classified as "denied"
    let mut max_accuracy_confusion_matrix = iterate_with_values(train_records, train_targets, test_records ,test_targets , 0.01, 100);
    let mut best_threshold = 0.0; // best_threshold is the threshold value that gives the highest accuracy
    let mut best_max_iterations = 0;
    let mut threshold = 0.02;

    for max_iterations in (1000..5000).step_by(500) {
        // max_iterations is the maximum number of iterations that the algorithm will run for before stopping.
        //here we are iterating over 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500 and 5000 iterations  on the training dataset.
        // step_by() is a method of the Range type that allows us to specify the step size between each iteration.
        // example:
        // for i in (1..10).step_by(2) {
        //     println!("{}", i);
        // }
        // will print 1, 3, 5, 7, 9

        // here the iteration will proceed until the threshold value is less than 0.1
        while threshold < 0.1 {
            let confusion_matrix = iterate_with_values(train_records, train_targets, test_records ,test_targets , threshold, max_iterations);

            // if the accuracy of the confusion matrix is greater than the accuracy of the max_accuracy_confusion_matrix
            if confusion_matrix.accuracy() > max_accuracy_confusion_matrix.accuracy() {
                max_accuracy_confusion_matrix = confusion_matrix;
                best_threshold = threshold;
                best_max_iterations = max_iterations;
            }
            threshold += 0.01; // incrementing the threshold value by 0.01
        }
        threshold = 0.02; // resetting the threshold value to 0.02
    }

    println!(
        "most accurate confusion matrix: {:?}",
        max_accuracy_confusion_matrix
    );
    println!(
        "with max_iterations: {}, threshold: {}",
        best_max_iterations, best_threshold
    );
    println!("accuracy {}", max_accuracy_confusion_matrix.accuracy(),); // accuracy is the number of correct predictions divided by the total number of predictions
    println!("precision {}", max_accuracy_confusion_matrix.precision(),); // precision is the number of true positives divided by the total number of true positives and false positives
    println!("recall {}", max_accuracy_confusion_matrix.recall(),); // recall is the number of true positives divided by the total number of true positives and false negatives
}

fn iterate_with_values(
   /*  train: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >,
    test: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >, */
    train_records: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    train_targets: ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 1]>>,
    test_records: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    test_targets: ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 1]>>,

    threshold: f64,

    /*
    threshold is a value used to classify a data point into one of two categories,
    typically when working with binary classification problems. For example, in this case,
    it is used to decide whether an exam result is considered as "accepted" or "denied".
    If the probability output from the logistic regression model is above the threshold,
    it is classified as "accepted" otherwise it is classified as "denied".
    */
    max_iterations: u64,
    /*
    max_iterations is the maximum number of iterations that the algorithm will run for before stopping.
    An iteration refers to one pass over the training dataset. In the context of this code,
    it controls the number of times the logistic regression model is updated using the training data.
    The purpose of having a maximum number of iterations is to prevent the model from getting stuck in an infinite loop,
    or running for too long. If the model reaches maximum iterations, it will stop and return the best solution
    it has found so far.
    */
) -> ConfusionMatrix<&'static str> {


    let train = Dataset::new(train_records, train_targets);
    let test = Dataset::new(test_records, test_targets);


    let model = LogisticRegression::default()
        .max_iterations(max_iterations)
        .gradient_tolerance(0.0001) // will stop when the change in the loss between two iterations falls below 0.0001
        .fit(&train) // fit() is a method of LogisticRegression which is used to train the model on the given dataset.
        // finds the best parameters for the model using the training data.
        .expect("can fit/train model");

    let validation = model.set_threshold(threshold).predict(test); // we will make prediction on the actual test data
    // by a threshold  of 0.99, which means that if the probability output from the logistic regression model is above 0.99,
     // it is classified as "accepted" otherwise it is classified as "denied".

    let confusion_matrix = validation
        .confusion_matrix(&test)
        .expect("can create confusion matrix");

    confusion_matrix // confusion matrix can also be said to error matrix it actually shows the performance of the model
}

/* Gradient tolerance, also known as convergence tolerance, is a hyperparameter used in optimization algorithms.
It is used to control the stopping criterion of the optimization algorithm. The optimization algorithm stops when the
 change in the loss (or objective function) falls below the gradient tolerance.
In the code you provided, the optimization algorithm for logistic regression stops when the change in the loss
between two iterations falls below 0.0001. This means that if the change in the loss is less than or equal to 0.0001,
the algorithm will stop, regardless of the number of iterations.
This can be useful for preventing the algorithm from overfitting the data or running indefinitely if the loss
is not decreasing. */

fn load_data(path: &str) -> Dataset<f64, &'static str> {
    // Dataset is a struct inside prelude model having two generic types
    // f64 is a float type and &'static str is a string type, here &'static str is a reference to a string that is
    // allocated in read-only memory and is valid for the duration of the program.
    // str have static lifetime which means that they live for the entire duration of the program.
    // &'static str is a string slice, which is a reference to part of a String, and it has a static lifetime.

    // ReaderBuilder allow us to create a reader with a custom configuration.
    // it builds a csv reader instance which will get the data from the file at the given path.
    // has_headers() is a method of ReaderBuilder which is used to specify whether the CSV file has a header row.
    // header row is a row that contains the names of the columns in the CSV file.
    // we don't have header row in our file so we set it to false.
    // delimeter here means the character that separates the fields in the CSV file.
    // we have used comma as a delimeter.
    // from_path() is a method of ReaderBuilder which is used to create a reader from the given path.
    // expect() is a method of Result which is used to return the content of an Ok value or panic if the value is an Err.

    // we have used let mut reader because we are going to modify the reader.
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_path(path)
        .expect("can create reader");

    // declares a variable named array of type Array2<f64> which is a 2D array of float 64bit numbers type
    // from the ndarray crate and csv crate.
    // and initializes it with the value returned by the deserialize_array2_dynamic() method of Array2Reader.
    // deserialize_array2_dynamic() is a method of Array2Reader which is used to deserialize a 2D array from a CSV file
    // read from the reader Object created above.
    // It uses f64 type for the elements of the array and parses the elements from the CSV file.
    // expect() is a method of Result which is used to return the content of an Ok value or panic if the value is an Err.
    let array: Array2<f64> = reader
        .deserialize_array2_dynamic()
        .expect("can deserialize array");

    // two new variables named data and targets are declared and initialized with the values returned by the
    // slice() and column() methods of array respectively.
    // slice() is a method of Array2 which is used to return a view of the array with the given range.
    // s![] is a macro which is used to create a range of indices.
    // .. means all the indices.
    // 0..2 means the first two indices.
    // to_owned() is a method of ArrayView2 which is used to return an owned array with the same data as the view.
    // .to_owned() will basically return a new owned array having copy of the data in the view.
    // view of the array means  a reference to the array without copying the data
    // means that the data variable will contain the first all rows and columns except the last one.
    // column() will return the last column of the array.(1 day array) , containing all rows of the third column

    /* This is useful for example if the original array contains both the data points and the
    corresponding labels for a supervised learning task, you can separate them and use them for different purposes,
    like training and testing.
    It's worth noting that both data and targets here are now owned by their own variable and the original
    array could be dropped safely.
    */

    let (data, targets) = (
        array.slice(s![.., 0..2usize]).to_owned(), // what usize means ?  usize is a type of unsigned integer.
        array.column(2).to_owned().into_shape((array.shape()[0], 1)).unwrap(), // into_shape() is a method of ArrayBase which is used to reshape the array.
        // we are reshaping the target array because we want it to be a 2D array if not than there will be type mismatch error


        /* 
            It looks like the data variable is a two-dimensional array of type Array2<f64> while the targets variable is a one-dimensional array of type Array1<f64>
            In order to fix this, you could try to make sure that the data and targets have the same dimension before passing them to Dataset::new(data, targets).
            for example, you could use the reshape function to reshape the target as two-dimensional array
        */


        /* 
            
Here `array.slice(s![.., 0..2usize]).to_owned()` is to select the first two columns from the loaded 2D array from the csv file, this data is stored into data variable.

`array.column(2).to_owned()` is to select the 3rd column from the loaded array , this is the target of classification.

`array.column(2).to_owned().into_shape((array.shape()[0],1)).unwrap()` is trying to reshape the selected column, 1D array into 2D array with shape (row,1) so 
that it could match the number of rows in the data and can be passed as a target to new function.

The reshape function `into_shape((array.shape()[0],1))` takes a tuple as an argument, specifying the desired dimensions of the reshaped array. 
The first element of the tuple is the number of rows, and the second element is the number of columns. Here, it's reshaping the 1D array to have the same 
number of rows as the original array and 1 column

Please note the use of `.unwrap()` at the end is because of the into_shape method returns a Result type object , unwrap method is used to get the value 
from the result object if the result is OK or panic if the result is Error.

With these changes, the `data` and `targets` variables should have the same number of rows, and thus, should be able to be passed as arguments to the 
`Dataset::new(data, targets)` function without a type mismatch error.
        */
    );

    /*
        
    you need to provide a type annotation for the range that you are passing in the slice macro by adding 'usize' at the end of the range. for example, 
    you can change this
    array.slice(s![.., 0..2]).to_owned()       to
    array.slice(s![.., 0..2usize]).to_owned()
    This should tell Rust that you want the range to be of type usize, and it should be able to determine the correct type of the range.
    Additionally, you also need to make sure that all the other indices you pass to the macro are also of type usize.
        */

    // vec![] is a macro which is used to create a vector.
    // test 1 and test 2 are the names of the features., and are vector of string type.
    // feature_name is a variable of type Vec<&str> which is a vector of string slices.
    // which will be use to store the names of the features data .
    let feature_names = vec!["test 1", "test 2"];

    // dataset takes two arguments data and targets and returns a Dataset object.
    // initialzing the dataset object with the data and targets.
    // map_targets() is a method of Dataset which is used to map the targets to a new value.
    // if the target is 1 then it will return "accepted" else it will return "denied".
    // with_features_names() is a method chain of Dataset which is used to set the names of the features.

    // |x| is a closure which is used to capture the value of x.
    // closure is a function that can capture the enclosing environment.
    // here x is a reference to the target value.
    // x is coming from the targets variable.
    Dataset::new(data, targets)
        .map_targets(|x| -> &'static str { // returning the closure value because static lifetime is required otherwise we will get an error of lifetime mismatch.
            // why map_targets() is used here?
            // here .map_target() will map the value of every x from target array and make it either
            // "accepted" or "denied" and return it.

            // map_targets() is used to map the targets to a new value.
            //example:
            //  let dataset = Dataset::new(data, targets);
            //  let dataset = dataset.map_targets(|x| x + 1); // x+1 is the new value of the target.
            //  let dataset = dataset.map_targets(|x| x * 2);
            //  let dataset = dataset.map_targets(|x| x * 3);
            // the above code will multiply the targets by 6.

            if *x as usize == 1 {
                // *x means the value of x. dereferencing the value of x.
                "accepted"
            } else {
                "denied"
            }
        })
        .with_feature_names(feature_names)
    // at the end feature_names will be the name of the features of the dataset.
    // it will be returned as a Dataset object with test 1 test 2    target
}

// creating a scatter plot using plotlib

fn plot_data(
    // passed by reference so we can modify the data without copying it.
    // // DatasetBase is given to us by the linfa crate.
    // ArrayBase is given to us by the ndarray crate, it is a multiDimensional array.
    // it is defining that n-dimensional array of type f64 and string slices, where ownedRepr is the representation of the array
    // in memory., Dim here represents the dimension of the array.
    // inshort we are   defining that the dataset is a two 2D arrays of float 64bit numbers type and string slices.
    // one for  data and one for targets.
    train: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, // records or data
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>, // targets
    >,
) {
    // declares and initializes a new vector of tuples of float 64bit numbers.
    //+ve the vector will contain the data of the accepted students.
    //-ve the vector will contain the data of the denied students.
    let mut positive = vec![]; // the type is recognized through the context.
    let mut negative = vec![];

    // records() is a method of DatasetBase which is used to return a reference to the records.
    // clone() is a method of ArrayBase which is used to return a new array with a copy of the data.
    // creates a deep copy of the array , means that create a new array with the same elements
    // but with a different memory location.
    //useful when you want to modify the array without modifying the original array.

    // into_raw_vec method consumes array and returns 1D vector with the same elements in the same order as they
    //appear in memory, can be useful when we want to perform other low level operation on the array

    // chunks() is a method of Vec which is used to return an iterator over the slice in chunks of size n.
    // here n is 2, so it will return an iterator over the slice in chunks of size 2.
    // here the slice is the records of the dataset.
    // collect() is a method of Iterator which is used to collect the iterator into a collection.
    // here the collection is a vector of slices of float 64bit numbers.

    let records = train.records().clone().into_raw_vec();
    let features: Vec<&[f64]> = records.chunks(2).collect();
    let targets = train.targets().clone().into_raw_vec();

    // features.len would  return the length of the features vector.
    //example:
    //      let a = vec![1,2,3,4,5];
    //      a.len() will return 5.
    // get() is a method of Vec which is used to return a reference to the element at the given index.
    for i in 0..features.len() {
        // feature should exist if not we panic.
        let feature = features.get(i).expect("feature exists"); // feature is a slice of float 64bit numbers.
        // feature is a reference to the reference of the slice of float 64bit numbers.

        // here we are checking if the target is "accepted" or "denied".
        // Some is an enum which is used to represent an optional value.
        // Some(x) represents a value of x.
        // None represents a missing value.
        if let Some(&"accepted") = targets.get(i) {
            positive.push((feature[0], feature[1])); // tuple of type Vec of float 64bit numbers., they are the numbers of the accepted students.
        } else {
            negative.push((feature[0], feature[1])); // numbers of the denied students.
        }
    }

    // creating a scatter plot using plotlib.

    // creates a new plot object and initializes it with the data of the accepted students(data in postiive vector)
    let plot_positive = Plot::new(positive)
        // point_style() is a method of Plot which is used to set the style of the points in the plot.
        .point_style(
            PointStyle::new() // creates a new PointStyle object.
                .size(2.0) // sets the size of the points in the plot.
                .marker(PointMarker::Square) // sets the marker of the points in the plot.
                .colour("#00ff00"), // sets the color of the points in the plot.
        )
        .legend("Exam Results".to_string()); // legend means   the name of the plot.
                                             // legend will be shown   in the top right corner of the plot.

    let plot_negative = Plot::new(negative).point_style(
        PointStyle::new()
            .size(2.0)
            .marker(PointMarker::Circle)
            .colour("#ff0000"),
    );

    // creating a grid for the plot. grid means the lines which are used to divide the plot into different sections.
    let grid = Grid::new(0, 0);

    //is used to organize and display a plot that is made up of continuous data.
    /* It is used to group multiple plot layers, set the axis labels and ranges, and control the overall appearance of
    the plot. It also allows to plot different set of data on the same view with different labels, markers, and colors.
    It creates a single plot with multiple layers and it can be customized according to the requirement,
    like setting the range of x and y axis and labels, and grid properties to improve the readability and it
    can be saved as an SVG file. */

    let mut image = ContinuousView::new() // ContinuousView is a struct which is used to create a continuous view of the plot
        .add(plot_positive)
        .add(plot_negative)
        .x_label("Exam 1 Score")
        .y_label("Exam 2 Score")
        .x_range(0.0, 120.0)
        .y_range(0.0, 120.0);

    image.add_grid(grid); // adds the grid to the plot.

    Page::single(&image) // creates a new page with the image.
        .save("plot.svg") // saves the plot as a svg file.
        .expect("can generate svg for plot")
}

/*
prelude is a module which contain most commonly used functions,type and trait.
It is common for crates to have a prelude module that re-exports commonly used items so
that users of the crate can easily access them without having to navigate the crate's module hierarchy.
The * (asterisk) symbol is a wildcard, and in this case it is used to import all items from the prelude module.
This is a common way to import everything from a module to make it easier to use its contents without having to
specify them individually.
*/

/*
Note that the deserialize_array2_dynamic() method is dynamic in the sense that it automatically adapts to the
number of rows and columns in the CSV file, this means that the number of rows and columns does not
need to be specified upfront, it will read them all into the array. Also, if the number of fields
are not the same in all rows, it will return an error.
*/
