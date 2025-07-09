with import <nixpkgs> {};
pkgs.mkShell {

  nativeBuildInputs = [ pkgs.bashInteractive ];

  buildInputs = with pkgs; [
    uv

  ];

}
