import glob
import hashlib

#import cv2
import datetime
import json
from os.path import join

from pandas import DataFrame
from typing import List, Iterator, Tuple, Optional

import numpy as np
from tqdm import tqdm
import pandas as pd

#from src.utils.util import check_directory
import time

class DatasetJson:
    def __init__(self, dataset_name : str, classes : List[str], images : List, ignore_wrong_dataset_name : bool =False):
        """
        This constructor should not be used, use the class mehtods from Definition and from File
        :param dataset_name:
        :param classes:
        :param images:
        """
        self.dataset_name = dataset_name
        self.classes = classes
        self.images = images
        self.ignore_wrong_dataset_name = ignore_wrong_dataset_name

        # create images dictionary
        self.images_dict = {entry['path'] : entry for entry in self.images}

        assert len(self.images_dict) == len(self.images), "Found different number of entries %d vs. %d, " \
                                                          "check for duplicate paths" \
                                                          % ( len(self.images_dict),len(self.images))

        # print("init with %d" % len(self.images))
        
    @classmethod
    def from_definition(cls, dataset_name, classes, ignore_wrong_dataset_name = False):
        """
        use ignore wrong dataset only when you know what you are doing, this will lead to a ẃrongly formated dataset json
        :param dataset_name:
        :param classes:
        :param ignore_wrong_dataset_name:
        :return:
        """
        return DatasetJson(dataset_name, classes, [], ignore_wrong_dataset_name)

    @classmethod
    def from_file(cls, file_path):
        """
        load dataset json
        :param file_path:
        :return:
        """

        with open(file_path,"r") as file:
            json_content = json.load(file)

            # ensure elements are in json
            assert "name" in json_content, "Wrong json contents: %s" % json_content
            assert "classes" in json_content, "Wrong json contents: %s" % json_content
            assert "images" in json_content, "Wrong json contents: %s" % json_content

            return DatasetJson(json_content["name"], json_content["classes"], json_content["images"])

    def add_image(self, path, split, cl, info=None):

        # check that correct values are provided
        if not self.ignore_wrong_dataset_name:
            assert path.startswith(self.dataset_name), "Expected %s at beginning of %s" % (self.dataset_name, path)
        assert cl in self.classes, "Expected %s to be in %s" % (cl, self.classes)

        if info is None:
            info = {}

        self.images.append(
            {"path": path, "split": split, "gt": cl, 'info': info}
        )

    def get_image_iterator(self) -> Iterator[Tuple[str,str,str]]:
        """
        Create an iterator for the images in this dataset,
        :return: iterator of tuples path (beginning with dataset name), dataset split and gt class (must be in classes)
        """
        for img_entry in self.images:
            yield (img_entry['path'], img_entry['split'], img_entry['gt'])


    def get_infos(self, path):
        return self.images_dict[path]['info']


    def save_jsons(self, out_path):

        dataset_json = {"name": self.dataset_name, "classes": self.classes, "images": self.images}

        check_directory(out_path, delete_old=False)

        # print("dump to file...")
        # print(len(self.images))
        with open(join(out_path, '%s.json' % self.dataset_name), 'w') as outfile:
            json.dump(dataset_json, outfile)

class PredictionJson:
    def __init__(self, config, ID_experiment, jsons):
        # self.dataset = config.dataset

        if config is not None:
            # print(config, type(config))
            if not isinstance(config,str):
                config = dict(vars(config))
                config_json = json.dumps({i: config[i] for i in config if i != 'graph'})
            else:
                config_json = config
        else:
            config_json = "Not available"
        self.config_json = config_json

        self.jsons = jsons
        self.ID_experiment = ID_experiment # identifies the experiment the results belong to, should be unqiue



    def add_predictions(self, sub_name, file_names, predictions, confidences, class_labels, clustering=False):
        """

        :param sub_name: identifies values from different elements belonging to the one experiment, e.g. different runs or evaluation methodss
        :param file_names:
        :param predictions:
        :param confidences:
        :param class_labels:
        :param clustering: boolean that indicates if the values are based on a clustering or classficiation
        :return:
        """

        assert len(predictions) == len(file_names), "list should have the same length %d vs. %d" % ( len(predictions),len(file_names))
        assert len(file_names) == len(confidences),  "list should have the same length %d vs. %d" % ( len(file_names),len(confidences))

        items = []
        # print(confidences)
        for i in range(len(file_names)):
            file = file_names[i]
            prediction= predictions[i]
            confidence = confidences[i]

            if clustering:
                item = {"image_path": file, "cluster_label": int(prediction), "class_label": None, "confidence": confidence.item()}
            else:
                item = {"image_path": file, "class_label": class_labels[prediction], "cluster_label": -1, "confidence": confidence.item()}

            items.append(item)

        prediction_yaml = {"config": self.config_json, "created_at": time.strftime("%d-%m-%Y-%H-%M-%S"),
                           "identifier": self.ID_experiment,
                           "name": sub_name, "predictions": items,
                           # "dataset": self.dataset
                           }

        self.jsons.append(prediction_yaml)

    def save_json_list(self, out_path):
        with open(join(out_path, 'predictions-%s.json' % (self.ID_experiment)), 'w') as outfile:
            json.dump(self.jsons, outfile)

    def get_prediction_iterator(self) -> Iterator[Tuple[str, str, str, datetime.datetime, str, int, str, float]]:
        """
        return iterator over predictions with config as json, identifier, name,  created_at, image_path, cluster_label, class_label, confidence
        :return:
        """

        for json in self.jsons:
            for pred in json['predictions']:
                yield json['config'], json['identifier'], json['name'], json['created_at'],  pred['image_path'], pred['cluster_label'], pred['class_label'], pred.get('confidence',-1)

    def get_general_info_for_set(self,index):
        json = self.jsons[index]

        return  json['config'], json['identifier'], json['name'], json['created_at']
    @property
    def length(self):
        return len(self.jsons)

    def get_iterator_for_set(self, index) -> Iterator[Tuple[str, int, str, float]]:
        """
        Get iterator based on the index of the internal sets, image_path, cluster_label, class_label, confidence
        :param index:
        :return:
        """

        json = self.jsons[index]
        for pred in json['predictions']:
            yield  pred['image_path'], pred['cluster_label'], pred['class_label'], pred.get('confidence',-1)


    @classmethod
    def from_definition(cls, config, ID_experiment):
        return PredictionJson(config,ID_experiment, [])

    @classmethod
    def from_file(cls, prediction_file):
        with open(prediction_file, 'r') as file:
            jsons = json.load(file)

            assert len(jsons) > 0, "empty file found"

            return PredictionJson(jsons[0]['config'], cls.get_identifier_from_file(prediction_file), jsons)

    @classmethod
    def get_identifier_from_file(cls, file_name):
        """
        get identifier from raw file name, take last element from directory seperators and cut predicitons and .json
        :param file_name:
        :return:
        """
        return file_name.split("/")[-1].split("predictions-")[-1].split(".json")[0]


class AnnotationJson:
    def __init__(self, dataset_name, anno_table : Optional[DataFrame], anno_json):
        """

        The Annotation Aggregation can either be init with an table of annotations, pd_dataframe, filename x classes, and a user who created the annotations
        or by a list of Annotation lists

        :param dataset_name: name of the dataset
        :param anno_table:
        :param anno_json:
        """

        self.dataset_name = dataset_name

        self.anno_table : DataFrame = anno_table
        self.anno_json = anno_json



    @classmethod
    def from_pandas_table(cls, dataset_name, anno_table, user, id_to_filename_format="%s", set_name ="generated"):
        """

        :param anno_table: filename x classes,
        :param user:
        :param id_to_filename_format: format string to convert the id to a file_name
        :return:
        """

        # case 2: annotation table is provided
        assert anno_table is not None
        assert user is not None
        # -> create an json list

        print("convert annotation table to list")

        ids = list(anno_table.index.values)
        labels_for_dataset = list(anno_table.columns)

        # cast table of annotations to a list of annotations per image, mind that not all images have the same number of annotations
        all_annotations = []
        now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        for id in tqdm(ids):

            annotations_per_object = 0
            for label in labels_for_dataset:
                number_annotations = int(anno_table.loc[id, label])

                for i in range(number_annotations):
                    # index = i + annotations_per_object
                    # annotations = all_annotations[index] if index in all_annotations else []
                    all_annotations.append(
                        {"image_path": id_to_filename_format % id, "class_label": label,
                          "created_at": now})

                    # all_annotations[index] = annotations

                annotations_per_object += number_annotations

        json_annotations = []
        # for key in all_annotations:
        name = "%s-%s" % (dataset_name, set_name)

        json_anno = {"name": name, "user_mail": user, "annotation_time": 0.0,"dataset_name": dataset_name,
                     "annotations": all_annotations}
        json_annotations.append(json_anno)

        return AnnotationJson(dataset_name, anno_table, json_annotations)

    @classmethod
    def from_file(cls, annotation_file):
        """
        Load annotation file and return handler
        :param annotation_file:
        :return:
        """

        with open(annotation_file, 'r') as outfile:

            annotation_jsons = json.load(outfile)

            # case 1: annotation json list

            assert len(annotation_jsons) > 0 , "File  must contain at least one annotations set to loadable otherwise dataset can't be determined"

            dataset_name = annotation_jsons[0]['dataset_name']

            # -> create table
            print("convert annotation jsons to table")
            # get names and labels
            img_names = []
            labels = []

            for annos in annotation_jsons:
                for entry in annos["annotations"]:

                    # add only valid annotations to table
                    if entry["class_label"] is not None:

                        # if entry["image_path"] not in img_names:
                        img_names.append(entry["image_path"])

                        # if entry["class_label"] not in img_names:
                        labels.append(entry["class_label"])

            img_names = list(np.unique(np.array(img_names)))
            labels = list(np.unique(np.array(labels)))

            # fast access maps
            map_names = dict(zip(img_names, list(np.arange(0, len(img_names)))))
            map_labels = dict(zip(labels, list(np.arange(0, len(labels)))))

            data = np.zeros((len(img_names), len(labels)))

            for annos in tqdm(annotation_jsons):
                for entry in annos["annotations"]:
                    # add only valid annotations to table
                    if entry["class_label"] is not None:
                        data[map_names[entry["image_path"]], map_labels[entry["class_label"]]] += 1


            return AnnotationJson(dataset_name, pd.DataFrame(data=data, index=img_names, columns=labels), annotation_jsons)

    @classmethod
    def from_predictions(cls, dataset_name, predictions_files : List[str], time=datetime.datetime.now(), ignore_wrong_dataset_name=False):
        """
        Load predicition file and return handler, will only use annotations which match the dataet
        :param predictions_file:
        :return:
        """

        annotation_json = AnnotationJson.empty(dataset_name)

        for predictions_file in predictions_files:
            print("load predictions file %s" % predictions_file)
            prediction_json = PredictionJson.from_file(predictions_file)

            for i in tqdm(range(prediction_json.length)):
                predictions = [ (pred[0], pred[2], time)
                                for pred in prediction_json.get_iterator_for_set(i)
                                if pred[2] is not None
                                and (ignore_wrong_dataset_name or pred[0].startswith(dataset_name))
                                ]

                if len(predictions) > 0:
                    annotation_json.add_annotationset("predicted_anno_%s_%s" % (prediction_json.jsons[i]['identifier'], prediction_json.jsons[i]['name']),
                                                      "import_user@mail.com", 0.0, predictions, ignore_wrong_dataset=ignore_wrong_dataset_name)

        return annotation_json



    @classmethod
    def empty(cls, dataset_name):
        """
        Be aware that no pandas table is used
        :param dataset_name:
        :return:
        """
        return AnnotationJson(dataset_name, None, [])


    def map_to_new_directory(self, dataset_name, original_directory,  new_directory):
        """
        Will discard any images which are not in the directory and the annotation
        :param original_directory: give a directory which was used to create these annotations
        :param new_directory: give a directory to which the annotations should be mapped, must end with / and only .png images are used
        :return:
        """

        print("map to new directory ...")

        # save and reset internals
        original_anno_json = self.anno_json.copy()
        self.anno_json = []
        self.anno_table = None
        self.dataset_name = dataset_name

        assert new_directory.endswith(dataset_name + "/"), "By convention the new directory should be folder of dataset" \
                                                           " and the name should be the dataset_name " \
                                                           "and directory should end with /," \
                                                           " but %s and %s was given" % (dataset_name, new_directory)

        for anno_set in original_anno_json:
            print("parse set %s " % anno_set['name'])

            # calculate hash values for original annotations
            # dict with hashes as key and list of images as value
            print("generated hashes ...")
            hash_map = {}
            for anno_entry in tqdm(anno_set['annotations']):
                old_img_path = anno_entry["image_path"]
                load_path = join(original_directory, old_img_path)
                img = cv2.imread(load_path)
                if img is not None:
                    hash = hashlib.md5(img).hexdigest()
                    list_of_old_paths = hash_map.get(hash,[])
                    list_of_old_paths.append((old_img_path, anno_entry['class_label'], anno_entry['created_at']))
                    hash_map[hash] = list_of_old_paths
                else:
                    print("WARNING:  Image %s not found" % (load_path))

            print("generated %d hashes" % len(hash_map.keys()))

            # iterate over new images and check if entry exists
            # create table of new entries
            print("search in new directory ...")
            all_files = sorted(glob.glob(new_directory + '/**/*.png', recursive=True))
            print("found %d files" % len(all_files))
            new_annotations = []
            for file in tqdm(all_files):
                img = cv2.imread(file)
                hash = hashlib.md5(img).hexdigest()
                if hash in hash_map:
                    # found imag
                    new_img_path = join(dataset_name,file.split(new_directory)[-1]) # add dataset_name
                    for old_img_path, class_label, created_at in hash_map[hash]:
                        # print("%s -> %s" % (old_img_path, new_img_path))
                        new_annotations.append((new_img_path, class_label, datetime.datetime.strptime(created_at, "%d-%m-%Y-%H-%M-%S")))

            print("reidentified %d images in new directory" % len(new_annotations))

            # set internal values to values calculated by
            # reset time because it might not be valid anymore
            self.add_annotationset(anno_set["name"], anno_set["user_mail"], 0.0 , new_annotations)


    def add_annotationset(self, anno_set_name, user, annotation_time, annotations : List[Tuple[str,str, datetime.datetime]], update_summary_table=True, ignore_wrong_dataset=False):
        """
        add a new annotation set
        :param anno_set_name:
        :param user:
        :param annotation_time:
        :param annotations: list of tuples with, imagepath, class_label and timestamp of creation (either datetime or string), class_label can be none
        :param update_summary_table: update the summary table, if you dont update, this will lead to incosistency if it is accessed later
        :return:
        """

        set_json = {
            "name": anno_set_name, "dataset_name": self.dataset_name,
            "user_mail": user, "annotation_time": annotation_time, "annotations": []
        }
        for image_path, class_label, created_at in tqdm(annotations):

            assert image_path.startswith(self.dataset_name) or ignore_wrong_dataset, "By convention the image path %s should start with the dataset_name %s" % (image_path, self.dataset_name)

            if class_label is None:
                continue # ignore this element


            anno_json = {"image_path": image_path,
                         "created_at": created_at.strftime("%d-%m-%Y-%H-%M-%S") if not isinstance(created_at, str) else created_at,
                         "class_label": class_label}

            set_json["annotations"].append(anno_json)


            if update_summary_table:
                # add to summary table
                if self.anno_table is None:
                    self.anno_table = pd.DataFrame({class_label: []})

                if class_label not in self.anno_table.columns:
                    self.anno_table[class_label] = 0
                if image_path not in self.anno_table.index:
                    # add new row
                    # cols = list(self.anno_table.columns)
                    # ids = list(self.anno_table.index.values) + [image_path]
                    # self.anno_table = self.anno_table.append(pd.DataFrame(data=np.zeros((len(cols),1)), index=ids, columns=cols))
                    self.anno_table.loc[image_path] = 0



                # increae number
                self.anno_table.loc[image_path, class_label] += 1


        # add to list of annotations
        self.anno_json.append(set_json)


    def get_annotation_iterator(self) -> Iterator[Tuple[str, str, float, str, datetime.datetime, str]]:
        """
        Create an iterator for the annotation set
        :return: iterator of tuples anno_set_name, dataset_name, user, annotation_time, image_path, created_at, class_label
        """
        for anno_set in self.anno_json:
            for anno_entry in anno_set['annotations']:
                yield (anno_set["name"], anno_set["dataset_name"], anno_set["user_mail"], anno_set["annotation_time"], anno_entry["image_path"], anno_entry["created_at"], anno_entry["class_label"])

    def get_annotationsets(self) -> Iterator[Tuple[str, str, float, List[Tuple[str,str, datetime.datetime]]]]:
        """
        iterate over th annotation sets
        """

        for anno_set in self.anno_json:
            annos = []
            for anno_entry in anno_set['annotations']:
                annos.append((anno_entry["image_path"], anno_entry["class_label"], anno_entry["created_at"]))

            yield (anno_set["name"], anno_set["dataset_name"], anno_set["user_mail"], anno_set["annotation_time"], annos)


    def add_other_anno_file(self, annotationfile: str, ignore_wrong_dataset = False):
        """
        add another annotation file to this annotation expects to have the same dataset name
        :param annotationfile:
        :param ignore_wrong_dataset: during adding the new item ignores wrong dataset warnings
        :return:
        """
        temp = AnnotationJson.from_file(annotationfile)

        for name, dataset_name, user, time, annos in temp.get_annotationsets():
            self.add_annotationset(name,user, time, annos, ignore_wrong_dataset=ignore_wrong_dataset)

    def get_table(self):
        return self.anno_table

    def save_json_list(self, path, file_name_format = "%s.json"):
        """

        :param path:
        :param file_name_format: change the file format, needs one parameter %s to insert the dataset
        :return:
        """

        check_directory(path, delete_old=False)

        with open(join(path, file_name_format % (self.dataset_name)),
                  'w') as outfile:
            json.dump(self.anno_json, outfile)

    def get_probability_data(self):
        """
        get the probabilites form the underlying dataframe, and the classes, names
        :return:
        """

        if self.anno_table is None:
            return [], [], np.array([])

        # convert annotations to probabilies
        classes = list(self.anno_table.columns.values)
        imgs = list(self.anno_table.index.values)
        data = self.anno_table.to_numpy()

        data = data / data.sum(axis=1, keepdims=True)  # cast to probs

        return imgs, classes, data

