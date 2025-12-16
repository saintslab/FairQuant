import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from fairquant.datasets import get_dataloaders
from fairquant.models import get_model
from fairquant.utils import set_seed
from train import train_one_epoch, evaluate 


def main():
    parser = argparse.ArgumentParser(description="Pre-train a full-precision model based on FairPrune paper.")
    parser.add_argument("--dataset", type=str, default="fitzpatrick17k",
                        choices=["celeba", "fitzpatrick17k", "isic2019"])
    parser.add_argument("--target_attribute", type=str, default="Blond_Hair", help="The target attribute for CelebA classification.")
    parser.add_argument("--sensitive_attribute", type=str, default="Male", help="The sensitive attribute for CelebA fairness analysis.")
    parser.add_argument("--fitzpatrick_binary_grouping", action="store_true", help="If set, groups Fitzpatrick17k into light (1-3) and dark (4-6) skin tones.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet18",
        # choices=["resnet18", "resnet34", "resnet50", "vgg11", "vgg16", "vgg19"]  
        help="Model name. Supports torchvision and timm models (e.g., tiny_vit_5m_224)."
    )
    parser.add_argument("--epochs", type=int, default=200, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument("--image_size", type=int, default=128, help="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
    )
    parser.add_argument("--train_subset", type=float, default=None)
    parser.add_argument("--test_subset", type=float, default=None)
    parser.add_argument("--positive_class", type=int, default=None)
    args = parser.parse_args()
    set_seed(42)
    device = torch.device(args.device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root_abs = os.path.join(script_dir, args.data_root)

    train_loader, test_loader, num_classes, class_names, num_groups, group_names = get_dataloaders(
        dataset=args.dataset,
        data_root=data_root_abs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        target_attribute=args.target_attribute,
        sensitive_attribute=args.sensitive_attribute,
        fitzpatrick_binary_grouping=args.fitzpatrick_binary_grouping
    )


    model = get_model(args.model, num_classes=num_classes, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)
    
    print(f"Starting pre-training for {args.epochs} epochs on dataset {args.dataset} (FairPrune setup)...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        
        val_loss, val_groups = evaluate(model, test_loader, device, num_groups, num_classes, args.positive_class, compute_parity_gaps=False)
        
        print(
            f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} avg_acc={val_groups['overall']['avg_acc']:.3f} "
            f"worst_acc={val_groups['overall']['worst_acc']:.3f} acc_gap={val_groups['overall']['acc_gap']:.3f}"
        )
        
        scheduler.step() 
        
    output_dir = "./checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    filename_suffix = ""
    if args.dataset == "celeba":
        filename_suffix = f"_{args.target_attribute}"
    elif args.dataset == "fitzpatrick17k" and args.fitzpatrick_binary_grouping:
        filename_suffix = "_binary"

    save_path = os.path.join(output_dir, f"{args.model}_{args.dataset}{filename_suffix}_pretrained.pt")

    torch.save(model.state_dict(), save_path)
    print(f"\nPre-training complete. Model checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()