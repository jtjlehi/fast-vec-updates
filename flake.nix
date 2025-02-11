{
  description = "Rust development template using fenix";

  inputs = {
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    utils,
    fenix,
    ...
  }:
    utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [fenix.overlays.default];
        };
        toolchain = pkgs.fenix.complete;
      in rec
      {
        # Executed by `nix build`
        packages.default =
          (pkgs.makeRustPlatform {
            # Use nightly rustc and cargo provided by fenix for building
            inherit (toolchain) cargo rustc;
          })
          .buildRustPackage {
            pname = "template";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;
          };

        # Used by `nix develop`
        devShells.default = pkgs.mkShell {
          # Use nightly cargo & rustc provided by fenix. Add for packages for the dev shell here
          buildInputs = with pkgs; [
            (with toolchain; [
              cargo
              rustc
              rust-src
              clippy
              rustfmt
              rust-analyzer
            ])
            pkg-config
            # for plotting benchmark data with criterion.rs
            gnuplot
          ];

          # Specify the rust-src path (many editors rely on this)
          RUST_SRC_PATH = "${toolchain.rust-src}/lib/rustlib/src/rust/library";
        };
      }
    );
}
