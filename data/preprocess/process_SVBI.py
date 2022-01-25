import os
import glob
import random
import helpers


def idx_to_file(idx):
    return "/".join(idx.split("-")[:-1])



if __name__ == "__main__":

    dataset_names = ["SVBI"]

    data = []

    for name in dataset_names:

        root_dir = f"/media/zyang/Error-Driven-ASR-Personalization/datasets/l2arctic/{name}/"
        transcript_dir = os.path.join(root_dir, "transcript")
        wav_dir = os.path.join(root_dir, "wav")

        for filename in os.listdir(transcript_dir):
            file_id = filename.split('.')[0]

            # get text (transcript)
            with open(os.path.join(transcript_dir, file_id + '.txt')) as f:
                text = f.readlines()[0]

            wav_path = os.path.join(wav_dir, file_id + '.wav')
            data.append({"text": text, "audio_filepath": wav_path, "duration": helpers.measure_audio_duration(wav_path)})

    
        random.seed(123456)
        random.shuffle(data)

        path_to_store = os.path.join("../", name, 'manifests')
        
        os.makedirs(path_to_store, exist_ok=True)
        helpers.write_json_data(os.path.join(path_to_store, "all.json"), data)

        n = len(data)

        selection = ("selection", int(n * 0.75))
        seed = ("seed", int(n * 0.1))
        dev = ("dev", int(n * 0.1))
        test_size = n - selection[1] - seed[1] - dev[1]
        test = ("test", test_size)

        lower = 0
        
        for name, interval in [seed, dev, selection, test]:
            upper = lower + interval
            
            curr_data = data[lower:upper] 
            helpers.write_json_data(f"{path_to_store}/{name}.json", curr_data)
            
            lower = upper

        helpers.write_json_data(f"{path_to_store}/seed_plus_dev.json", data[0:(seed[1]+dev[1])])

