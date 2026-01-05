import os

def Get_path(data_name, prot = '1', sub_prot = None):
    # Get the project root directory (parent of data_process folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    Dir = os.path.join(project_root, 'datasets')

    train_image_dir, train_list = None, None
    val_image_dir, val_list = None, None
    test_image_dir, test_list = None, None

    if data_name == 'OULU-NPU': # no depth map
        # OULU-NPU: prot in ['1', '2', '3', '4'], sub_prot is None or ['1', '2', '3', '4', '5', '6']
        if prot == '1' or prot == '2':
            train_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Train.txt'
            val_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Dev.txt'
            test_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Test.txt'
        else:
            train_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Train_{sub_prot}.txt'
            val_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Dev_{sub_prot}.txt'
            test_list = f'{Dir}/OULU-NPU-1/Prot/Protocol_{prot}/Test_{sub_prot}.txt'

        train_image_dir = f'{Dir}/OULU-NPU-1/'
        val_image_dir = f'{Dir}/OULU-NPU-1/'
        test_image_dir = f'{Dir}/OULU-NPU-1/'
    

    elif data_name == 'CASIA':
        train_image_dir = Dir
        train_list = f"{Dir}/cbnData/prot/CASIA_train.txt"

        val_image_dir = Dir
        val_list = f"{Dir}/cbnData/prot/CASIA_val.txt"

        test_image_dir = Dir
        test_list = f"{Dir}/cbnData/prot/CASIA_test.txt"

    elif data_name == 'RA':
        train_image_dir = Dir
        train_list = f"{Dir}/cbnData/prot/RA_train.txt"

        val_image_dir = Dir
        val_list = f"{Dir}/cbnData/prot/RA_val.txt"

        test_image_dir = Dir
        test_list = f"{Dir}/cbnData/prot/RA_test.txt"

    elif data_name == 'MSU':
        train_image_dir = f"{Dir}/MSU-MFSD"
        train_list = f"{Dir}/MSU-MFSD/train_list.txt"

        val_image_dir = f"{Dir}/MSU-MFSD"
        val_list = f"{Dir}/MSU-MFSD/test_list.txt"

        test_image_dir = f"{Dir}/MSU-MFSD"
        test_list = f"{Dir}/MSU-MFSD/test_list.txt"

    elif data_name == 'SIW':
        # SIW: prot_sub_prot in ['1_1', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2']
        train_image_dir = f"{Dir}/SIW"
        train_list = f"{Dir}/SIW/train_list_{prot}_{sub_prot}.txt"

        val_image_dir = f"{Dir}/SIW"
        val_list = f"{Dir}/SIW/test_list_{prot}_{sub_prot}.txt"

        test_image_dir = f"{Dir}/SIW"
        test_list = f"{Dir}/SIW/test_list_{prot}_{sub_prot}.txt"

    elif data_name == 'CASIA-SURF':
        train_image_dir = f'{Dir}/CASIA-SURF/'
        train_list = f'{Dir}/CASIA-SURF/train_list.txt'

        val_image_dir = f'{Dir}/CASIA-SURF/'
        val_list = f'{Dir}/CASIA-SURF/val_private_list.txt'

        test_image_dir = f'{Dir}/CASIA-SURF/'
        test_list = f'{Dir}/CASIA-SURF/test_private_list.txt'

    elif data_name == 'WMCA':
        # prot1:rigidmask; prot2:replay;  prot3:prints;       prot4:papermask;
        # prot5:grandtest; prot6:glasses; prot7:flexiblemask; prot8:fakehead
        # RGB、Color、Depth、Infrared、Thermal
        NAME = 'WMCA-1'
        PROT = 'PORT2'
        train_image_dir = f'{Dir}/{NAME}/'
        train_list = f"{Dir}/{NAME}/{PROT}/{prot}_train_list.txt"

        val_image_dir = f'{Dir}/{NAME}/'
        val_list = f"{Dir}/{NAME}/{PROT}/{prot}_dev_list.txt"

        test_image_dir = f'{Dir}/{NAME}/'
        test_list = f"{Dir}/{NAME}/{PROT}/{prot}_test_list.txt"

    # 跨数据及测试CASIA-RA
    elif data_name == 'CASIA-RA':
        train_image_dir = Dir
        train_list = f"{Dir}/cbnData/prot/CASIA_train.txt"

        val_image_dir = Dir
        val_list = f"{Dir}/cbnData/prot/CASIA_val.txt"

        test_image_dir = Dir
        test_list =f"{Dir}/cbnData/prot/RA_test.txt"

    elif data_name == 'RA-CASIA':
        train_image_dir = Dir
        train_list = f"{Dir}/cbnData/prot/RA_train.txt"

        val_image_dir = Dir
        val_list = f"{Dir}/cbnData/prot/RA_val.txt"

        test_image_dir = Dir
        test_list = f"{Dir}/cbnData/prot/CASIA_test.txt"

    train_path = {'image_dir': train_image_dir, 'prot_list': train_list}
    val_path = {'image_dir': val_image_dir, 'prot_list': val_list}
    test_path = {'image_dir': test_image_dir, 'prot_list': test_list}

    return train_path, val_path, test_path