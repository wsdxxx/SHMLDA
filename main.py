import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from mask import Mask
from model import (
    EdgeDecoder,
    GMAE,
    SoftHGNNEncoder,
    load_state_dict_compat,
)
from utils import (
    get_data,
    make_undirected_edge_index,
    sample_bipartite_negatives,
    set_seed,
)


warnings.filterwarnings("ignore")


def split_train_validation(train_indices, validation_ratio, generator):
    if len(train_indices) < 2:
        raise ValueError("At least two training edges are required per fold.")

    permutation = torch.randperm(
        len(train_indices),
        generator=generator,
    ).numpy()
    validation_size = int(round(len(train_indices) * validation_ratio))
    validation_size = min(max(validation_size, 1), len(train_indices) - 1)

    validation_indices = train_indices[permutation[:validation_size]]
    training_indices = train_indices[permutation[validation_size:]]
    return training_indices, validation_indices


def metrics_dict(auc, ap, acc, sen, pre, spe, f1, mcc):
    return {
        "AUC": "{:.6f}".format(auc),
        "AP": "{:.6f}".format(ap),
        "ACC": "{:.6f}".format(acc),
        "SEN": "{:.6f}".format(sen),
        "PRE": "{:.6f}".format(pre),
        "SPE": "{:.6f}".format(spe),
        "F1": "{:.6f}".format(f1),
        "MCC": "{:.6f}".format(mcc),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic settings
    parser.add_argument("--dataset", type=int, default=1, choices=[1, 2, 3, 4], help="Choose dataset 1, 2, 3, or 4.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for model and dataset.")
    parser.add_argument("--dim", type=int, default=1000, help="Output feature dimension used by the encoder.")
    parser.add_argument("--save_tag", type=str, default="", help="Optional tag for the output subdirectory.")

    # Soft hypergraph settings
    parser.add_argument("--num_hyperedges", type=int, default=32, help="Number of soft hyperedges.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in soft hyperedge generation.")
    parser.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for hyperedge assignment.")
    parser.add_argument("--p", type=float, default=0.4, help="Mask ratio.")
    parser.add_argument("--ssm_expand", type=int, default=2, help="Expansion factor of the bidirectional selective SSM mixer.")
    parser.add_argument("--disable_ssm", action="store_true", help="Disable the bidirectional selective SSM hyperedge mixer.")
    parser.add_argument("--hyperedge_residual_scale", type=float, default=1.0, help="Residual scale from node features in the soft-hyperedge token mixer.")

    # Structure and decoder settings
    parser.add_argument("--disable_membership_fusion", action="store_true", help="Disable pair-level soft-hyperedge co-membership fusion in the edge decoder.")
    parser.add_argument("--use_structure_residual", action="store_true", help="Enable the legacy post-softmax dual-channel residual.")
    parser.add_argument("--disable_structural_conditioning", action="store_true", help="Disable structural priors in prototype and membership logits.")

    # Evaluation settings
    parser.add_argument("--validation_ratio", type=float, default=0.2, help="Fraction of each rotating 5-CV training fold reserved for validation.")
    parser.add_argument("--evaluation_protocol", type=str, default="5cv", choices=["5cv", "independent"], help="Use rotating 5-CV tests or one fixed independent test split.")

    # Training settings
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of training epochs.")
    parser.add_argument("--eval_every", type=int, default=10, help="Validate every N epochs.")
    args = parser.parse_args()
    if not 0.0 < args.p < 1.0:
        parser.error("--p must be strictly between 0 and 1.")
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("--validation_ratio must be strictly between 0 and 1.")
    if args.epochs <= 0 or args.eval_every <= 0:
        parser.error("--epochs and --eval_every must be positive.")
    if args.num_hyperedges <= 0 or args.num_heads <= 0:
        parser.error("--num_hyperedges and --num_heads must be positive.")
    if args.ssm_expand <= 0:
        parser.error("--ssm_expand must be positive.")
    if args.hyperedge_residual_scale < 0:
        parser.error("--hyperedge_residual_scale must be non-negative.")
    if 64 % args.num_heads != 0:
        parser.error("--num_heads must divide the hidden dimension (64).")
    set_seed(args.seed)

    data = get_data(args.dataset, args.dim)
    pos_edge = data.pos_edge_label_index.t().cpu().numpy()
    full_edge = data.pos_edge_label_index
    num_nodes = data.num_nodes
    num_lnc = int(data.num_lnc)
    num_dis = int(data.num_dis)

    print(
        f">>> Dataset D{args.dataset}: num_lnc={num_lnc}, "
        f"num_dis={num_dis}, num_nodes={num_nodes}"
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    if args.evaluation_protocol == "5cv":
        fold_specs = []
        for train_val_idx, test_idx in kf.split(pos_edge):
            split_generator = torch.Generator().manual_seed(
                args.seed + len(fold_specs)
            )
            train_idx, val_idx = split_train_validation(
                train_val_idx,
                args.validation_ratio,
                split_generator,
            )
            fold_specs.append((train_idx, val_idx, test_idx))

        fixed_test_pos = None
        fixed_test_neg = None
        result_scope = "Test"
    else:
        outer_splits = list(kf.split(pos_edge))
        cv_pool_idx = outer_splits[0][0]
        fixed_test_idx = outer_splits[0][1]

        inner_kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed,
        )
        fold_specs = []
        for train_pool_idx, val_idx in inner_kf.split(cv_pool_idx):
            fold_specs.append(
                (
                    cv_pool_idx[train_pool_idx],
                    cv_pool_idx[val_idx],
                    fixed_test_idx,
                )
            )

        fixed_test_pos = torch.tensor(pos_edge[fixed_test_idx]).t().long()
        fixed_test_generator = torch.Generator().manual_seed(args.seed)
        fixed_test_neg = sample_bipartite_negatives(
            num_lnc=num_lnc,
            num_dis=num_dis,
            num_samples=fixed_test_pos.size(1),
            exclude_edge_index=full_edge,
            generator=fixed_test_generator,
            disease_offset=num_lnc,
        )
        result_scope = "Independent Test"

        print(
            f">>> Independent test size: "
            f"{fixed_test_pos.size(1)} positives + "
            f"{fixed_test_neg.size(1)} negatives"
        )
        print(
            f">>> Remaining cross-validation pool: "
            f"{len(cv_pool_idx)} positive edges"
        )

    all_results = []
    fold_results_list = []

    best_overall_val_auc = -np.inf
    best_overall_model_state = None
    best_overall_relation_edge = None
    best_overall_fold = 0
    best_overall_epoch = 0

    for fold, (train_idx, val_idx, test_idx) in enumerate(fold_specs):
        print(f"\n========== Fold {fold + 1} ==========")
        fold_generator = torch.Generator().manual_seed(args.seed + fold)

        train_pos = torch.tensor(pos_edge[train_idx]).t().long()
        val_pos = torch.tensor(pos_edge[val_idx]).t().long()
        train_relation_edge = make_undirected_edge_index(train_pos)
        test_pos = (
            fixed_test_pos
            if fixed_test_pos is not None
            else torch.tensor(pos_edge[test_idx]).t().long()
        )

        # Test candidates are held out first. Validation and training
        # negatives can never reuse them.
        test_neg = (
            fixed_test_neg
            if fixed_test_neg is not None
            else sample_bipartite_negatives(
                num_lnc=num_lnc,
                num_dis=num_dis,
                num_samples=test_pos.size(1),
                exclude_edge_index=full_edge,
                generator=fold_generator,
                disease_offset=num_lnc,
            )
        )
        val_neg = sample_bipartite_negatives(
            num_lnc=num_lnc,
            num_dis=num_dis,
            num_samples=val_pos.size(1),
            exclude_edge_index=torch.cat((full_edge, test_neg), dim=1),
            generator=fold_generator,
            disease_offset=num_lnc,
        )
        train_neg = sample_bipartite_negatives(
            num_lnc=num_lnc,
            num_dis=num_dis,
            num_samples=train_pos.size(1),
            exclude_edge_index=torch.cat(
                (full_edge, test_neg, val_neg),
                dim=1,
            ),
            generator=fold_generator,
            disease_offset=num_lnc,
        )

        train_data = data.clone()
        train_data.edge_index = data.edge_index
        train_data.relation_edge_index = train_relation_edge
        train_data.mask_relation_structure = True
        train_data.pos_edge_label_index = train_pos
        train_data.neg_edge_label_index = train_neg

        out_dim = 128
        hidden_dim = 64
        encoder = SoftHGNNEncoder(
            data.x.size(1),
            hidden_dim,
            out_dim,
            args.num_hyperedges,
            args.num_heads,
            temperature=args.tau,
            structural_conditioning=(
                not args.disable_structural_conditioning
            ),
            structure_residual=args.use_structure_residual,
            use_ssm=not args.disable_ssm,
            ssm_expand=args.ssm_expand,
            hyperedge_residual_scale=args.hyperedge_residual_scale,
        )
        edge_decoder = EdgeDecoder(
            out_dim,
            hidden_dim,
            use_membership=not args.disable_membership_fusion,
            membership_dim=2 * args.num_hyperedges,
        )
        mask = Mask(p=args.p)
        model = GMAE(encoder, edge_decoder, mask).cpu()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=5e-5,
        )
        best_val_auc = -np.inf
        best_val_epoch = 0
        best_state = None

        for epoch in range(args.epochs):
            model.train()
            loss = model.train_epoch(
                train_data,
                optimizer,
                generator=fold_generator,
            )

            should_validate = (
                (epoch + 1) % args.eval_every == 0
                or epoch + 1 == args.epochs
            )
            if not should_validate:
                continue

            model.eval()
            with torch.no_grad():
                z = model.encoder(
                    data.x,
                    data.edge_index,
                    train_relation_edge,
                )
                val_metrics = model.test(z, val_pos, val_neg)

            val_auc = val_metrics[0]
            print(
                f"Epoch {epoch + 1:03d}, Loss: {loss:.6f}, "
                f"Val AUC: {val_auc:.6f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                print(
                    f"      [New Best] validation updated at "
                    f"epoch {epoch + 1}"
                )

        if best_state is None:
            raise RuntimeError(f"Fold {fold + 1} produced no validation state.")

        load_state_dict_compat(model, best_state, verbose=False)
        model.eval()
        with torch.no_grad():
            z = model.encoder(
                data.x,
                data.edge_index,
                train_relation_edge,
            )
            test_metrics = model.test(z, test_pos, test_neg)

        fold_result = metrics_dict(*test_metrics)
        all_results.append(fold_result)
        fold_results_list.append({**fold_result, "fold": fold + 1})

        print(
            f"Fold {fold + 1} Best Val AUC: {best_val_auc:.6f} "
            f"at epoch {best_val_epoch}"
        )
        print(f"Fold {fold + 1} {result_scope} Results: {fold_result}")

        if best_val_auc > best_overall_val_auc:
            best_overall_val_auc = best_val_auc
            best_overall_model_state = best_state
            best_overall_relation_edge = train_relation_edge.detach().cpu().clone()
            best_overall_fold = fold + 1
            best_overall_epoch = best_val_epoch

    for index, result in enumerate(all_results):
        print(f"Fold {index + 1} {result_scope} Results: {result}")

    average_results = {
        metric: np.mean([float(result[metric]) for result in all_results])
        for metric in all_results[0]
    }
    if args.evaluation_protocol == "independent":
        print(
            "Average Independent Test Results over 5 CV folds:",
            average_results,
        )
    else:
        print("Average Results over 5 folds:", average_results)

    df_folds = pd.DataFrame(fold_results_list)
    summary = {}
    for key in df_folds.columns:
        if key == "fold":
            continue
        values = df_folds[key].astype(float)
        summary[key + "_mean"] = values.mean()
        summary[key + "_std"] = values.std()
    df_summary = pd.DataFrame([summary])

    save_dir_run = (
        os.path.join(
            "results",
            args.save_tag or args.evaluation_protocol,
            f"dataset{args.dataset}",
        )
        if args.save_tag or args.evaluation_protocol == "independent"
        else os.path.join("results", f"dataset{args.dataset}")
    )
    os.makedirs(save_dir_run, exist_ok=True)

    fold_metrics_name = (
        "independent_fold_metrics.csv"
        if args.evaluation_protocol == "independent"
        else "fold_metrics.csv"
    )
    summary_metrics_name = (
        "independent_metrics_summary.csv"
        if args.evaluation_protocol == "independent"
        else "metrics_summary.csv"
    )
    df_folds.to_csv(
        os.path.join(save_dir_run, fold_metrics_name),
        index=False,
    )
    df_summary.to_csv(
        os.path.join(save_dir_run, summary_metrics_name),
        index=False,
    )

    if best_overall_model_state is not None:
        best_model_name = (
            "best_cv_model.pt"
            if args.evaluation_protocol == "independent"
            else "best_model.pt"
        )
        best_model_path = os.path.join(save_dir_run, best_model_name)
        torch.save(best_overall_model_state, best_model_path)
        relation_edge_path = os.path.join(
            save_dir_run,
            "best_relation_edge_index.pt",
        )
        torch.save(best_overall_relation_edge, relation_edge_path)
        print(
            f">>> Best CV model selected by validation AUC at "
            f"Fold {best_overall_fold}, Epoch {best_overall_epoch}"
        )
        print(f">>> Best model saved to: {best_model_path}")
        print(f">>> Best relation edges saved to: {relation_edge_path}")

    print(f">>> Results saved to: {save_dir_run}")
