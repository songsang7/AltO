import logging
import os
import time

import torch
import torchvision
import numpy as np

from common.utils import GeneralUtils, GeometryUtils
from common.controller.base_controller import BaseController
from common import lr_schedules
from common import optimizers
from image_matching.utils import ImageMatchingUtils
from image_matching.definitions import DataKey
from image_matching.networks import alto_net, dhn_net, raft, ihn_net, rhwf_net


class Alto(BaseController):
    """
    Ping-Pong 방식으로 r과 f를 학습
    """
    def __init__(self):
        super().__init__()

        # registrator
        self._net_r = None
        self._opt_r = None
        self._lr_r = None

        # feature extractor
        self._net_f = None
        self._opt_f = None
        self._lr_f = None

    def _load_best_models_weights(self):
        best_model_path_r = os.path.join(self._result_path, f"best_model_r.pt")
        self._net_r.load_state_dict(torch.load(best_model_path_r, map_location="cpu"))
        
        best_model_path_f = os.path.join(self._result_path, f"best_model_f.pt")
        self._net_f.load_state_dict(torch.load(best_model_path_f, map_location="cpu"))

    def _save_best_models(self):
        os.makedirs(self._result_path, exist_ok=True)

        best_model_path_r = os.path.join(self._result_path, f"best_model_r.pt")
        torch.save(self._net_r.state_dict(), best_model_path_r)

        best_model_path_f = os.path.join(self._result_path, f"best_model_f.pt")
        torch.save(self._net_f.state_dict(), best_model_path_f)

    def _save_last_models(self):
        os.makedirs(self._result_path, exist_ok=True)

        model_path_r = os.path.join(self._result_path, f"last_model_r.pt")
        torch.save(self._net_r.state_dict(), model_path_r)

        model_path_f = os.path.join(self._result_path, f"last_model_f.pt")
        torch.save(self._net_f.state_dict(), model_path_f)

    def _set_models(self, param):
        dataset = self._dataset_train if self._dataset_train is not None else self._dataset_test
        c, h, w = dataset.get_data_shape()

        # registrator
        self._net_r = self._set_model_r(c, h, w)

        # feature extractor
        self._net_f = alto_net.FeatureExtractor(c).to(self._device)

    def _set_model_r(self, c, h, w) -> torch.nn.Module:
        pass  # override

    def _set_opts_lrs(self, train_param):
        self._opt_r = optimizers.get_optimizer(self._net_r.parameters(), train_param["optimizer"], init_lr=train_param["lr"]["init_lr"], l2_regular=train_param["l2_regular"])
        self._lr_r = lr_schedules.get_lr_scheduler(self._opt_r, self._epoch_cnt, len(self._train_data_feeder), train_param["lr"])

        self._opt_f = optimizers.get_optimizer(self._net_f.parameters(), train_param["optimizer"], init_lr=train_param["lr"]["init_lr"], l2_regular=train_param["l2_regular"])
        self._lr_f = lr_schedules.get_lr_scheduler(self._opt_f, self._epoch_cnt, len(self._train_data_feeder), train_param["lr"])

    def _change_models_phase(self, is_training: bool):
        if is_training:
            self._net_r.train()
            self._net_f.train()
        else:
            self._net_r.eval()
            self._net_f.eval()

    def _train_batch(self, batch):
        self._change_models_phase(True)

        # (1) Update R network
        GeneralUtils.set_net_requires_grad(self._net_f, False)
        GeneralUtils.set_net_requires_grad(self._net_r, True)
        outputs_r, loss_r = self._feed_forward_r(batch)
        self._feed_backward_r(loss_r)

        # (2) Update F network
        GeneralUtils.set_net_requires_grad(self._net_f, True)
        GeneralUtils.set_net_requires_grad(self._net_r, False)
        outputs_f, loss_f = self._feed_forward_f(batch, outputs_r)
        self._feed_backward_f(loss_f)

        self._step_lr()

        with torch.no_grad():
            batch_metric = self._metric_func(batch, outputs_r, loss_r, loss_f)

        losses = [loss_r.detach().cpu(), loss_f.detach().cpu()]
        return losses, batch_metric.detach().cpu()

    @torch.no_grad()
    def _validate_batch(self, batch):
        self._change_models_phase(False)

        outputs_r, loss_r = self._feed_forward_r(batch)
        outputs_f, loss_f = self._feed_forward_f(batch, outputs_r)

        batch_metric = self._metric_func(batch, outputs_r, loss_r, loss_f)

        losses = [loss_r.detach().cpu(), loss_f.detach().cpu()]
        return losses, batch_metric.detach().cpu()

    def _feed_forward_r(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        pass  # override

    def _feed_backward_r(self, batch_loss_r):
        self._opt_r.zero_grad(set_to_none=True)
        batch_loss_r.backward()
        torch.nn.utils.clip_grad_norm_(self._net_r.parameters(), 1.0)
        self._opt_r.step()

    def _feed_forward_f(self, batch, outputs_r):
        image_patch_fixed = batch[DataKey.IMAGE_PATCH_FIXED].to(self._device)
        h_patch_pred, warped_moving = outputs_r

        f_f, z_f = self._net_f(image_patch_fixed, enc_only=False)
        f_m, z_m = self._net_f(warped_moving.detach(), enc_only=False)

        # loss = GeneralUtils.infonce_loss_1d(z_m, z_f)
        loss = GeneralUtils.bt_loss_1d(z_m, z_f)

        outputs = [f_f, z_f, f_m, z_m]
        return outputs, loss

    def _feed_backward_f(self, batch_loss_f):
        self._opt_f.zero_grad(set_to_none=True)
        batch_loss_f.backward()
        self._opt_f.step()

    def _step_lr(self):
        self._lr_r.step()
        self._lr_f.step()

    def _metric_func(self, batch, outputs_r, batch_loss_r, batch_loss_f):
        image_patch_moving, h_patch = batch[DataKey.IMAGE_PATCH_MOVING], batch[DataKey.H_PATCH]
        if h_patch is None:
            return batch_loss_r

        h_patch_pred, warped_moving = outputs_r

        height, width = image_patch_moving.shape[-2:]
        result = ImageMatchingUtils.calc_mace_3(height, width, h_patch_pred, h_patch.to(h_patch_pred.device))
        return result

    def _post_epoch_train(self, start_time, train_losses, train_metrics, valid_losses, valid_metrics) -> bool:
        avg_train_metric = sum(train_metrics) / len(train_metrics)

        train_losses = torch.tensor(train_losses)
        avg_train_loss_r, avg_train_loss_f = train_losses.mean(dim=0)
        monitoring_loss, monitoring_metric = avg_train_loss_r, avg_train_metric

        avg_valid_loss_r = avg_valid_loss_f = avg_valid_metric = None
        if 0 < len(valid_losses):
            avg_valid_metric = sum(valid_metrics) / len(valid_metrics)
            valid_losses = torch.tensor(valid_losses)
            avg_valid_loss_r, avg_valid_loss_f = valid_losses.mean(dim=0)
            monitoring_loss, monitoring_metric = avg_valid_loss_r, avg_valid_metric

        is_new_best = self._check_best_model(monitoring_metric)
        if is_new_best:
            self._save_best_models()

        self._print_epoch_wise_msg(start_time, avg_train_loss_r, avg_train_loss_f, avg_train_metric, avg_valid_loss_r, avg_valid_loss_f, avg_valid_metric, is_new_best)
        return False  # False : continue train
    
    def _post_epoch_test(self, start_time, test_losses, test_metrics):
        avg_test_metric = sum(test_metrics) / len(test_metrics)
        test_losses = torch.tensor(test_losses)
        avg_test_loss_r, avg_test_loss_f = test_losses.mean(dim=0)

        end_time = time.time()
        duration_time = end_time - start_time
        # time_stamp = GeneralUtils.get_time_stamp(end_time)

        msg = f"[Test] [Test Loss_(R, F):({avg_test_loss_r:.4f}, {avg_test_loss_f:.4f})] [Test Metric:{avg_test_metric:.4f}] - {duration_time:.2f}s"
        logging.info(msg)

    def _print_epoch_wise_msg(self, start_time, avg_train_loss_r, avg_train_loss_f, avg_train_metric, avg_valid_loss_r, avg_valid_loss_f, avg_valid_metric, is_new_best):
        end_time = time.time()
        duration_time = end_time - start_time

        # time_stamp = GeneralUtils.get_time_stamp(end_time)

        valid_msg = "[No Validation]"
        if avg_valid_loss_r is not None:
            valid_msg = f"[Valid Loss_(R, F):({avg_valid_loss_r:.4f}, {avg_valid_loss_f:.4f})] [Valid Metric:{avg_valid_metric:.4f}]"

        msg = f"[Epoch:{self._cur_epoch}] [Train Loss_(R, F):({avg_train_loss_r:.4f}, {avg_train_loss_f:.4f})] [Train Metric:{avg_train_metric:.4f}] {valid_msg} - {duration_time:.2f}s"
        if is_new_best:
            msg += " - best_model"

        logging.info(msg)

    @torch.inference_mode()
    def _print_result(self, print_param):
        os.makedirs(self._result_path, exist_ok=True)

        max_sample_cnt = 20
        self._change_models_phase(False)

        prefixes = ["train", "valid", "test"]
        data_feeders = [self._train_data_feeder, self._valid_data_feeder, self._test_data_feeder]

        for prefix, data_feeder in zip(prefixes, data_feeders):
            if data_feeder is None:
                continue

            q = len(data_feeder) // max_sample_cnt
            for idx, batch in enumerate(data_feeder):
                if idx % q != 0:
                    continue

                outputs_r, loss_r = self._feed_forward_r(batch)
                h_patch_gt = batch[DataKey.H_PATCH]
                h_patch_pred, warped_moving = outputs_r

                image_patch_moving, image_patch_fixed = batch[DataKey.IMAGE_PATCH_MOVING], batch[DataKey.IMAGE_PATCH_FIXED]
                image_patch_moving = image_patch_moving.to(h_patch_pred.device)
                image_patch_fixed = image_patch_fixed.to(h_patch_pred.device)

                warped_moving = GeometryUtils.get_batch_warped_image_from_homography(image_patch_moving, h_patch_pred)

                torchvision.utils.save_image(image_patch_moving[0], os.path.join(self._result_path, f"{prefix}_{idx}_moving.png"))
                torchvision.utils.save_image(image_patch_fixed[0], os.path.join(self._result_path, f"{prefix}_{idx}_fixed.png"))
                torchvision.utils.save_image(warped_moving[0], os.path.join(self._result_path, f"{prefix}_{idx}_warped_moving.png"))

                # print h matrix
                np.savetxt(os.path.join(self._result_path, f"{prefix}_{idx}_h_gt.txt"), h_patch_gt[0].detach().cpu().numpy())
                np.savetxt(os.path.join(self._result_path, f"{prefix}_{idx}_h_pred.txt"), h_patch_pred[0].detach().cpu().numpy())


class AltoDhn(Alto):
    def __init__(self):
        super().__init__()

    def _set_model_r(self, c, h, w) -> torch.nn.Module:
        return dhn_net.CnnRegistrator(c, h, w).to(self._device)

    def _feed_forward_r(self, batch):
        image_patch_moving, image_patch_fixed = batch[DataKey.IMAGE_PATCH_MOVING], batch[DataKey.IMAGE_PATCH_FIXED]

        image_patch_moving = image_patch_moving.to(self._device)
        image_patch_fixed = image_patch_fixed.to(self._device)

        _, h_patch_pred = self._net_r(image_patch_moving, image_patch_fixed)
        warped_moving = GeometryUtils.get_batch_warped_image_from_homography(image_patch_moving, h_patch_pred)

        f_f, _ = self._net_f(image_patch_fixed, enc_only=True)
        f_m, _ = self._net_f(warped_moving, enc_only=True)

        # loss = GeneralUtils.infonce_loss_2d(f_f, f_m)
        loss = GeneralUtils.bt_loss_2d(f_m, f_f)
        outputs = [h_patch_pred, warped_moving]
        return outputs, loss


class AltoRaft(Alto):
    def __init__(self):
        super().__init__()

    def _set_model_r(self, c, h, w) -> torch.nn.Module:
        return raft.RaftLarge(c).to(self._device)

    def _feed_forward_r(self, batch):
        image_patch_moving, image_patch_fixed = batch[DataKey.IMAGE_PATCH_MOVING], batch[DataKey.IMAGE_PATCH_FIXED]

        image_patch_moving = image_patch_moving.to(self._device)
        image_patch_fixed = image_patch_fixed.to(self._device)

        flow_field_list = self._net_r(image_patch_moving, image_patch_fixed, False)

        height, width = image_patch_moving.shape[-2:]
        grid = GeometryUtils.gen_2d_grid_torch(height, width).to(self._device)
        total_loss = 0.0
        warped_moving = None
        k = len(flow_field_list)
        for i in range(k):
            w = 0.8 ** (k - i - 1)
            h_patch_pred_i = GeometryUtils.get_batch_homography_from_flow_field_sub(grid, flow_field_list[i])
            warped_moving = GeometryUtils.get_batch_warped_image_from_homography(image_patch_moving, h_patch_pred_i)

            f_f, _ = self._net_f(image_patch_fixed, enc_only=True)
            f_m, _ = self._net_f(warped_moving, enc_only=True)

            # loss_i = GeneralUtils.infonce_loss_2d(f_f, f_m)
            loss_i = GeneralUtils.bt_loss_2d(f_m, f_f)
            total_loss += w * loss_i

        h_patch_pred = GeometryUtils.get_batch_homography_from_flow_field_sub(grid, flow_field_list[-1].detach())
        outputs = [h_patch_pred, warped_moving]
        return outputs, total_loss


class AltoIhn(Alto):
    def __init__(self):
        super().__init__()

    def _set_model_r(self, c, h, w) -> torch.nn.Module:
        return ihn_net.IHNNet(c, h, w, False).to(self._device)

    def _feed_forward_r(self, batch):
        image_patch_moving, image_patch_fixed = batch[DataKey.IMAGE_PATCH_MOVING], batch[DataKey.IMAGE_PATCH_FIXED]

        image_patch_moving = image_patch_moving.to(self._device)
        image_patch_fixed = image_patch_fixed.to(self._device)

        predictions = self._net_r(image_patch_moving, image_patch_fixed)
        disp_list, h_patch_pred = predictions

        height, width = image_patch_moving.shape[-2:]
        total_loss = 0.0
        warped_moving = None
        k = len(disp_list)
        for i in range(k):
            w = 0.85 ** (k - i - 1)
            h_patch_pred_i = GeometryUtils.get_batch_homography_from_corners_offset(height, width, disp_list[i])
            warped_moving = GeometryUtils.get_batch_warped_image_from_homography(image_patch_moving, h_patch_pred_i)

            f_f, _ = self._net_f(image_patch_fixed, enc_only=True)
            f_m, _ = self._net_f(warped_moving, enc_only=True)

            # loss_i = GeneralUtils.infonce_loss_2d(f_f, f_m)
            loss_i = GeneralUtils.bt_loss_2d(f_m, f_f)
            total_loss += w * loss_i

        outputs = [h_patch_pred, warped_moving]
        return outputs, total_loss


class AltoRhwf(Alto):
    def __init__(self):
        super().__init__()

    def _set_model_r(self, c, h, w) -> torch.nn.Module:
        return rhwf_net.RHWFNet(c, h, w, False).to(self._device)

    def _feed_forward_r(self, batch):
        image_patch_moving, image_patch_fixed = batch[DataKey.IMAGE_PATCH_MOVING], batch[DataKey.IMAGE_PATCH_FIXED]

        image_patch_moving = image_patch_moving.to(self._device)
        image_patch_fixed = image_patch_fixed.to(self._device)

        predictions = self._net_r(image_patch_moving, image_patch_fixed)
        disp_list, h_patch_pred = predictions

        height, width = image_patch_moving.shape[-2:]
        total_loss = 0.0
        warped_moving = None
        k = len(disp_list)
        for i in range(k):
            w = 0.85 ** (k - i - 1)
            h_patch_pred_i = GeometryUtils.get_batch_homography_from_corners_offset(height, width, disp_list[i])
            warped_moving = GeometryUtils.get_batch_warped_image_from_homography(image_patch_moving, h_patch_pred_i)

            f_f, _ = self._net_f(image_patch_fixed, enc_only=True)
            f_m, _ = self._net_f(warped_moving, enc_only=True)

            # loss_i = GeneralUtils.infonce_loss_2d(f_f, f_m)
            loss_i = GeneralUtils.bt_loss_2d(f_m, f_f)
            total_loss += w * loss_i

        outputs = [h_patch_pred, warped_moving]
        return outputs, total_loss
