import os
import time
import argparse

import torch
import torch.utils.data as data

import model
import config
import metrics
import data_utils
import json
import logger


def parser_add_argument	( parser ) :
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="batch size for training")
    parser.add_argument("--top_k",
                        type=int,
                        default=10,
                        help="compute metrics@top_k")
    parser.add_argument("--gpu",
                        type=str,
                        default="0",
                        help="gpu card ID")
    parser.add_argument("--dataset",
                        type=str,
                        default="valid",
                        help="train for 'valid' or 'test'")
    parser.add_argument("--factor_num",
                        type=int,
                        default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers",
                        type=int,
                        default=3,
                        help="number of layers in MLP model")
    parser.add_argument("--out",
                        default=True,
                        help="save model or not")
    return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = parser_add_argument(parser)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    start_time = time.time()

    #####################PREPARE DATA###########################
    print("Start predict.py, dataset : {}".format(args.dataset))
    logger.write_log(config.pred_log, "strart to predict file".format(args.epochs))

    train_data, test_question, test_answer, user_num ,item_num, train_mat, user_map, item_map = data_utils.load_all(args.dataset)

    inv_user_map = {v: k for k, v in user_map.items()}
    inv_item_map = {v: k for k, v in item_map.items()}

    test_dataset = data_utils.NCFData(
        test_question, item_num, train_mat, 0, False, user_map, item_map)
    test_loader = data.DataLoader(test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    ########################LOAD MODEL###########################
    GMF_model = None
    MLP_model = None

    model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model, GMF_model, MLP_model)
    model.load_state_dict(torch.load('{}{}_{}.pth'.format(config.model_path, config.model, args.dataset)))
    model.cuda()

    ##########################PREDICT#############################
    test_loader.dataset.all_sample_test()

    if args.dataset == 'test' :
            predictions = metrics.predict(model, test_loader, test_question, inv_user_map, inv_item_map, args.top_k)
    elif args.dataset == 'valid' :
            predictions = metrics.predict(model, test_loader, test_question, inv_user_map, inv_item_map, args.top_k)

    if args.out :
        if not os.path.exists(config.pred_path):
                os.mkdir(config.pred_path)

        if args.dataset == 'test':
                with open(os.path.join(config.pred_path, "pred.txt"), "w") as fp :
                        fp.write(json.dumps(predictions))
        if args.dataset == 'valid' :
                with open(os.path.join(config.pred_path, "pred_val.txt"), "w") as fp :
                        fp.write(json.dumps(predictions))

    elapsed_time = time.time() - start_time
    logger.write_log(config.pred_log, "The time elapse is: " +
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    logger.write_log(config.pred_log, "-------------------------------------")
